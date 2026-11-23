"""Planner, conversation, editor, and chat API tests. No RC1 load."""
import asyncio
import json
import os
import unittest
from pathlib import Path
os.environ["DHARM_SKIP_ENGINE"] = "1"
os.environ["GEMINI_EDITOR"] = "0"
os.environ.pop("GEMINI_API_KEY", None)

from fastapi.testclient import TestClient

from app.conversation import detect_query_intent, formulate_retrieval_query, retrieval_wording
from app.editor import accept_edit, editor_enabled
from app.planner import DecisionPlan, Support, build_plan, support_words
from app.render import render_plan
from app.server import SESSIONS, app


def load_records():
    path = Path("decision-corpus-ml/data/decisions_v0.2.jsonl")
    return {json.loads(line)["id"]: json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()}


RECORDS = load_records()


def history_from(user, reply, intent="new_query"):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": reply, "metadata": {"intent": intent}},
    ]


class Conversation(unittest.TestCase):
    def test_search_rewrite_stays_out_of_wording_and_funds_stay_funds(self):
        feature = "My team disagrees about releasing a feature. We have a week to decide."
        self.assertIn("software feature", retrieval_wording(feature))
        self.assertIn("releasing a feature", feature)
        self.assertEqual(retrieval_wording("release unused funds"), "release unused funds")
        funds = "We have surplus funds from last quarter. Should I release the unused budget back to the company, or keep it for the team?"
        self.assertEqual(detect_query_intent(funds, []), "new_query")

    def test_followup_keeps_decision_and_new_topic_drops_it(self):
        history = history_from(feature := "My team disagrees about releasing a feature. We have a week to decide.", "Meet first.")
        follow = "What if the deadline moves to this afternoon?"
        self.assertEqual(detect_query_intent(follow, history), "followup")
        query = formulate_retrieval_query(follow, history, "followup")
        self.assertIn("software feature", query)
        self.assertIn("afternoon", query)
        self.assertNotIn("supersedes", follow)
        jobs = "Should I change jobs?"
        self.assertEqual(detect_query_intent(jobs, history), "clarify")
        self.assertNotIn("feature", formulate_retrieval_query(jobs, history, "clarify"))

    def test_help_greeting_and_unnamed_it(self):
        history = history_from("We are releasing a feature.", "Talk first.")
        self.assertEqual(detect_query_intent("help", history), "clarify")
        self.assertEqual(detect_query_intent("help me decide", history), "clarify")
        self.assertEqual(detect_query_intent("Help me choose whether to apologize to my sister", history), "new_query")
        self.assertEqual(detect_query_intent("Should I release it tomorrow?", []), "clarify")
        self.assertEqual(detect_query_intent("Should I release the feature tomorrow?", []), "new_query")
        self.assertEqual(detect_query_intent("What about the other one?", history), "followup")
        self.assertEqual(detect_query_intent("Let's do the first thing you suggested", history), "followup")
        referred = formulate_retrieval_query("Let's do the first thing you suggested", history, "followup")
        self.assertIn("Talk first.", referred)


class Planner(unittest.TestCase):
    def test_feature_then_afternoon_then_jobs(self):
        first = "My team disagrees about releasing a feature. We have a week to decide. What should I do first?"
        plan = build_plan(first, [], "new_query", {
            "abstained": False, "primary": "D004", "top3": ["D004", "D001", "D032"],
            "pair_promoted": True, "promoted_pair": ["D004", "D001"],
        }, RECORDS)
        text = render_plan(plan)
        self.assertEqual(plan.disposition, "answer")
        self.assertIn("team", text.casefold())
        self.assertIn("week", text.casefold())
        self.assertTrue(plan.second_action)
        self.assertNotIn("[D", text)
        self.assertNotIn("misconduct", text.casefold())
        self.assertNotIn("What should I do first", text)
        self.assertEqual(plan.supports[0].id, "D004")

        history = history_from(first, text)
        follow = "What if the deadline moves to this afternoon?"
        revised = build_plan(follow, history, "followup", {
            "abstained": False, "primary": "D001", "top3": ["D001", "D010", "D022"],
            "pair_promoted": False, "promoted_pair": None,
        }, RECORDS)
        revised_text = render_plan(revised)
        self.assertEqual(revised.disposition, "clarify")
        self.assertIn("remedy", revised_text.casefold())
        self.assertIn("this afternoon", revised_text.casefold())
        self.assertIn("releasing a feature", revised.decision)
        self.assertNotIn("impossible", revised.action.casefold())

    def test_surplus_funds_are_not_a_software_release(self):
        message = "We have surplus funds from last quarter. Should I release the unused budget back to the company, or keep it for the team?"
        plan = build_plan(message, [], "new_query", {
            "abstained": False, "primary": "D014", "top3": ["D014", "D011", "D002"],
            "pair_promoted": False,
        }, RECORDS)
        text = render_plan(plan)
        self.assertEqual(plan.disposition, "answer")
        self.assertEqual(plan.supports[0].id, "D014")
        self.assertNotIn("disposable", plan.action.casefold())
        self.assertIn("surplus", text.casefold())
        self.assertIn("need", plan.limit.casefold())

    def test_personal_weak_and_correction(self):
        sister = "Help me choose whether to apologize to my sister before dinner tonight."
        plan = build_plan(sister, [], "new_query", {
            "abstained": False, "primary": "D014", "top3": ["D014", "D004", "D020"],
            "pair_promoted": False,
        }, RECORDS)
        self.assertEqual(plan.disposition, "answer")
        self.assertIn("sister", render_plan(plan).casefold())

        hair = build_plan("What color should I dye my hair?", [], "new_query", {
            "abstained": True, "primary": "Abstain", "top3": ["D041", "D051", "D069"],
            "pair_promoted": False,
        }, RECORDS)
        self.assertEqual(hair.disposition, "abstain")

        history = history_from(
            "The whole team must decide the launch date. We have a week.",
            "Hear both sides.",
        )
        correction = "Actually, it is just me and one teammate, not the whole team."
        self.assertEqual(detect_query_intent(correction, history), "followup")
        revised = build_plan(correction, history, "followup", {
            "abstained": False, "primary": "D004", "top3": ["D004", "D001", "D032"],
            "pair_promoted": True, "promoted_pair": ["D004", "D001"],
        }, RECORDS)
        self.assertIn("one teammate", revised.decision + revised.reason + " " + " ".join(revised.facts))
        self.assertIn("just me and one teammate", " ".join(revised.facts).casefold())


class Regressions(unittest.TestCase):
    def retrieval(self, primary, top3, promoted=False, pair=None):
        return {
            "abstained": False, "primary": primary, "top3": top3,
            "pair_promoted": promoted, "promoted_pair": pair,
        }

    def test_deadline_corrections_both_directions_and_two_events(self):
        today = "The release decision is today. What should we do first?"
        asked = build_plan(today, [], "new_query", self.retrieval("D001", ["D001", "D010", "D022"]), RECORDS)
        self.assertEqual(asked.disposition, "clarify")
        self.assertIn("remedy", asked.action.casefold())

        history = [
            {"role": "user", "content": today},
            {"role": "assistant", "content": asked.action, "metadata": {
                "intent": "clarify", "clarify_kind": "missing_fact",
                "plan": {"decision": today},
            }},
        ]
        restored = "Actually, it isn't today anymore. We have a week."
        self.assertEqual(detect_query_intent(restored, history), "followup")
        plan = build_plan(restored, history, "followup", self.retrieval("D001", ["D001", "D010", "D022"]), RECORDS)
        text = render_plan(plan)
        self.assertEqual(plan.disposition, "answer")
        self.assertEqual(plan.supports[0].id, "D001")
        self.assertIn("week", text.casefold())
        self.assertNotIn("impossible", plan.action.casefold())
        self.assertTrue(any("week" in fact and "today" in fact for fact in plan.facts))

        both = "The meeting is today, but the release decision is next week."
        split = build_plan(both, [], "new_query", self.retrieval("D001", ["D001", "D010", "D022"]), RECORDS)
        self.assertEqual(split.disposition, "answer")
        self.assertEqual(split.supports[0].id, "D001")
        self.assertIn("The meeting is today", render_plan(split))
        self.assertEqual(render_plan(split).casefold().count("the meeting is today"), 1)
        expiry_primary = build_plan(
            "The decision is today.", [], "new_query",
            self.retrieval("D010", ["D010", "D001"]), RECORDS,
        )
        self.assertEqual(expiry_primary.disposition, "clarify")
        self.assertTrue(any(fact.startswith("meeting: today") for fact in split.facts))
        self.assertTrue(any("next week" in fact for fact in split.facts))
        self.assertNotIn("impossible", split.action.casefold())

    def test_new_topic_then_clarification_then_answer_and_flip_direction(self):
        feature = "My team disagrees about releasing a feature. We have a week to decide."
        history = history_from(feature, "Hear both sides.")
        jobs = build_plan("Should I change jobs?", history, "clarify", self.retrieval("D032", ["D032"]), RECORDS)
        self.assertEqual(jobs.disposition, "clarify")
        self.assertIn("leaving your current job", jobs.action)
        self.assertNotIn("feature", render_plan(jobs).casefold())
        self.assertFalse(any("week" in fact for fact in jobs.facts))

        answered = [
            {"role": "user", "content": "Should I change jobs?"},
            {"role": "assistant", "content": jobs.action, "metadata": {
                "intent": "clarify", "clarify_kind": "missing_fact",
                "plan": {"decision": "Should I change jobs?"},
            }},
        ]
        reason = "I don't trust my manager and the pay is lower."
        self.assertEqual(detect_query_intent(reason, answered), "followup")
        used = build_plan(reason, answered, "followup", self.retrieval("D032", ["D032", "D033"]), RECORDS)
        self.assertEqual(used.disposition, "answer")
        self.assertIn("jobs", used.decision.casefold())
        self.assertIn("manager", render_plan(used).casefold())
        self.assertNotIn("feature", render_plan(used).casefold())

        remedy = "Waiting would close the only rollback remedy."
        prior = [
            {"role": "user", "content": feature},
            {"role": "assistant", "content": "Hear both sides.", "metadata": {"intent": "new_query"}},
        ]
        flipped = build_plan(remedy, prior, "followup", self.retrieval("D001", ["D001", "D010", "D022"]), RECORDS)
        self.assertEqual(flipped.supports[0].id, "D010")
        self.assertIn("open", flipped.limit.casefold())
        self.assertNotIn("cutoff", flipped.limit.casefold())

    def test_opposing_condition_word_is_not_positive_support(self):
        self.assertNotIn("cutoff", support_words(RECORDS["D001"]))
        self.assertIn("cutoff", (RECORDS["D001"]["competing_passage_id"]["flip_to_competing_when"]).casefold())

    def test_editor_stays_off_without_an_explicit_switch(self):
        os.environ["GEMINI_API_KEY"] = "present"
        os.environ.pop("GEMINI_EDITOR", None)
        try:
            self.assertFalse(editor_enabled())
        finally:
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ["GEMINI_EDITOR"] = "0"


class Editor(unittest.TestCase):
    def plan(self):
        return DecisionPlan(
            disposition="answer",
            decision="My team disagrees about releasing a feature.",
            action="Invite each willing party to describe the disputed behavior.",
            reason="This is about the team disagreement. The deadline to use is a week.",
            supports=[Support("D004", "Invite each willing party.", "primary")],
            limit="Verified misconduct requires a report.",
        )

    def test_invented_source_and_reversed_action_are_rejected(self):
        plan = self.plan()
        self.assertFalse(accept_edit(plan, "Wait, and cite [D999] as the authority."))
        self.assertFalse(accept_edit(plan, "Do not invite each willing party. Postpone the conversation."))
        kept = accept_edit(plan, "Your team still disagrees, and a week remains to hear both accounts.")
        self.assertTrue(kept)
        self.assertIn("Invite each willing party", render_plan(plan))
        held = DecisionPlan(
            disposition="answer",
            decision="Should we release the feature?",
            action="Hold the release until tests pass.",
            reason="Tests have not passed.",
            supports=[Support("D022", "Hold the release until tests pass.", "primary")],
        )
        self.assertFalse(accept_edit(held, "Shipping immediately is the correct choice."))
        self.assertFalse(accept_edit(held, "The client already approved a 4 million budget."))
        self.assertTrue(accept_edit(held, "Tests have not passed, so the release stays on hold."))


class ChatApi(unittest.TestCase):
    def test_answer_without_gemini_and_unavailable_editor(self):
        class Engine:
            def predict(self, query):
                return {
                    "top3": ["D004", "D001"],
                    "abstained": False,
                    "primary": "D004",
                    "pair_promoted": True,
                    "promoted_pair": ["D004", "D001"],
                }
        app.state.engine = Engine()
        app.state.records = {pid: RECORDS[pid] for pid in ("D004", "D001", "D010")}
        app.state.model_lock = asyncio.Lock()
        client = TestClient(app)
        try:
            response = client.post("/api/chat", json={
                "message": "My team disagrees about releasing a feature. We have a week to decide. What should I do first?",
                "session_id": "api-test",
            })
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual(events[0]["disposition"], "answer")
            self.assertEqual(events[0]["provider_mode"], "Dharm")
            self.assertFalse(events[0]["edited"])
            self.assertIn("week", events[1]["text"].casefold())
            self.assertNotIn("qwen", events[1]["text"].casefold())
            self.assertNotIn("[D", events[1]["text"])
            self.assertNotIn("misconduct", events[1]["text"].casefold())
            self.assertNotIn("**", events[1]["text"])
            self.assertTrue(events[0]["cited_ids"])
            self.assertEqual(events[-1]["status"], "complete")
        finally:
            SESSIONS.pop("api-test", None)
            client.close()

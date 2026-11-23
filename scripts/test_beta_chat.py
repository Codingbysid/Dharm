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
from app.editor import accept_edit
from app.planner import DecisionPlan, Support, build_plan
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
        self.assertEqual(detect_query_intent(jobs, history), "new_query")
        self.assertNotIn("feature", formulate_retrieval_query(jobs, history, "new_query"))

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
        self.assertTrue(plan.limit)

        history = history_from(first, text)
        follow = "What if the deadline moves to this afternoon?"
        revised = build_plan(follow, history, "followup", {
            "abstained": False, "primary": "D001", "top3": ["D001", "D010", "D022"],
            "pair_promoted": False, "promoted_pair": None,
        }, RECORDS)
        revised_text = render_plan(revised)
        self.assertEqual(revised.disposition, "answer")
        self.assertIn("this afternoon", revised_text.casefold())
        self.assertIn("deadline is now", revised_text.casefold())
        self.assertNotIn("feature release", revised.decision.casefold() + " ")  # decision may still name the feature
        self.assertIn("releasing a feature", revised.decision)
        self.assertIn("this afternoon", revised.facts[0])
        self.assertIn("replacing a week", revised_text.casefold())

        jobs = build_plan("Should I change jobs?", history, "new_query", {
            "abstained": False, "primary": "D032", "top3": ["D032", "D033", "D094"],
            "pair_promoted": False,
        }, RECORDS)
        jobs_text = render_plan(jobs)
        self.assertEqual(jobs.disposition, "abstain")
        self.assertNotIn("feature", jobs_text.casefold())
        self.assertNotIn("software", jobs_text.casefold())
        self.assertIn("change jobs", jobs_text.casefold())

    def test_surplus_funds_are_not_a_software_release(self):
        message = "We have surplus funds from last quarter. Should I release the unused budget back to the company, or keep it for the team?"
        plan = build_plan(message, [], "new_query", {
            "abstained": False, "primary": "D014", "top3": ["D014", "D011", "D002"],
            "pair_promoted": False,
        }, RECORDS)
        text = render_plan(plan)
        self.assertEqual(plan.disposition, "answer")
        self.assertIn("disposable", plan.action.casefold())
        self.assertIn("surplus", text.casefold())
        self.assertTrue(plan.limit)

    def test_personal_weak_and_correction(self):
        sister = "Help me choose whether to apologize to my sister before dinner tonight."
        plan = build_plan(sister, [], "new_query", {
            "abstained": False, "primary": "D014", "top3": ["D014", "D004", "D020"],
            "pair_promoted": False,
        }, RECORDS)
        self.assertEqual(plan.disposition, "answer")
        self.assertIn("sister", render_plan(plan).casefold())

        hair = build_plan("What color should I dye my hair?", [], "new_query", {
            "abstained": False, "primary": "D041", "top3": ["D041", "D051", "D069"],
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
            self.assertTrue(events[0]["cited_ids"])
            self.assertEqual(events[-1]["status"], "complete")
        finally:
            SESSIONS.pop("api-test", None)
            client.close()

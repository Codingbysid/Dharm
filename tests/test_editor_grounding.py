"""Tests for source-grounded Gemini second reader and editor guardrails."""
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from app.planner import DecisionPlan, Support
from app.render import render_plan
from app.editor import (
    accept_edit,
    build_evidence_bundle,
    editor_enabled,
    review_grounded_reply,
    verify_evidence_spans,
)

# Sample corpus records for testing
MOCK_RECORDS = {
    "D004": {
        "id": "D004",
        "work": "Arthashastra",
        "locator": "1.15.5",
        "english_translation": "Let the king hear both parties in the dispute before deciding.",
        "derived_principle": "Invite each willing party to describe the disputed behavior.",
        "application_limit": "Verified misconduct requires a report.",
    },
    "D001": {
        "id": "D001",
        "work": "Mahabharata",
        "locator": "5.34.12",
        "english_translation": "One should first establish the constraints of time and agreement.",
        "derived_principle": "Record the constraints each party must meet.",
        "application_limit": "A firm cutoff leaves no room for further negotiation.",
    },
    "D022": {
        "id": "D022",
        "work": "Manusmriti",
        "locator": "8.12",
        "english_translation": "Where justice is destroyed by injustice, the judges themselves are struck down.",
        "derived_principle": "Hold the release until tests pass.",
        "application_limit": "Routine non-blocking warnings do not warrant halting delivery.",
    },
}


class TestEditorGrounding(unittest.TestCase):
    def setUp(self):
        self.plan = DecisionPlan(
            disposition="answer",
            decision="My team disagrees about releasing a software feature.",
            action="Invite each willing party to describe the disputed behavior.",
            reason="Hearing both accounts preserves team consensus. We have a week to decide.",
            supports=[
                Support("D004", "Invite each willing party to describe the disputed behavior.", "primary")
            ],
            limit="Verified misconduct requires a report.",
            facts=["team disagrees on release", "deadline is one week"],
        )

    def test_network_spy_gemini_follows_the_key(self):
        """A loaded key turns the editor on. GEMINI_EDITOR=0 makes no HTTP call."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_secret_key", "GEMINI_EDITOR": "0"}):
            self.assertFalse(editor_enabled())
            with patch("httpx.post") as mock_post:
                res = review_grounded_reply(self.plan, MOCK_RECORDS)
                self.assertIsNone(res)
                mock_post.assert_not_called()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_secret_key"}, clear=False):
            os.environ.pop("GEMINI_EDITOR", None)
            self.assertTrue(editor_enabled())
            with patch("httpx.post") as mock_post:
                review_grounded_reply(self.plan, MOCK_RECORDS)
                mock_post.assert_called()

    def test_evidence_bundle_separation(self):
        """Bundle must separate classical text from modern editorial interpretation and limit."""
        bundle = build_evidence_bundle(self.plan, MOCK_RECORDS)
        self.assertEqual(len(bundle), 1)
        step = bundle[0]
        self.assertEqual(step["passage_id"], "D004")
        self.assertEqual(step["classical_text"], "Let the king hear both parties in the dispute before deciding.")
        self.assertEqual(step["modern_editorial_interpretation"], "Invite each willing party to describe the disputed behavior.")
        self.assertEqual(step["application_limit"], "Verified misconduct requires a report.")
        self.assertIn("deadline is one week", step["relevant_user_facts"])

    def test_span_grounding_rejects_hallucinated_evidence(self):
        """Spans not appearing verbatim in the corpus passage must be rejected."""
        valid_claims = [
            {
                "claim": "Both parties must be heard before making a decision.",
                "passage_id": "D004",
                "evidence_span": "hear both parties in the dispute",
            }
        ]
        self.assertTrue(verify_evidence_spans(valid_claims, self.plan, MOCK_RECORDS))

        hallucinated_claims = [
            {
                "claim": "Consensus is mandatory in corporate bylaws.",
                "passage_id": "D004",
                "evidence_span": "consensus is mandatory in corporate bylaws",
            }
        ]
        self.assertFalse(verify_evidence_spans(hallucinated_claims, self.plan, MOCK_RECORDS))

        unapproved_passage_claims = [
            {
                "claim": "A completely invented citation.",
                "passage_id": "D999",
                "evidence_span": "some invented text",
            }
        ]
        self.assertFalse(verify_evidence_spans(unapproved_passage_claims, self.plan, MOCK_RECORDS))

    def test_reversed_advice_rejected(self):
        """Explanations reversing or negating the approved action must be rejected."""
        # Contrasting verbs
        self.assertFalse(accept_edit(self.plan, "Skip the conversation and decide on your own."))
        self.assertFalse(accept_edit(self.plan, "Unilaterally decide the outcome without consulting."))
        self.assertFalse(accept_edit(self.plan, "Do not invite each willing party to discuss."))

        held_plan = DecisionPlan(
            disposition="answer",
            decision="Should we deploy the release?",
            action="Hold the release until tests pass.",
            reason="Tests are failing.",
            supports=[Support("D022", "Hold the release until tests pass.", "primary")],
        )
        self.assertFalse(accept_edit(held_plan, "Shipping immediately is the best decision."))
        self.assertFalse(accept_edit(held_plan, "Proceed now and ignore the test results."))

    def test_invented_facts_and_numbers_rejected(self):
        """New external numbers/statistics not present in user facts or corpus must be rejected."""
        self.assertFalse(accept_edit(self.plan, "A study showed that 85% of software teams disagree on releases."))
        self.assertFalse(accept_edit(self.plan, "The project budget of $450000 requires quick action."))

        # Numbers present in the plan's facts/reason are acceptable
        valid_plan = DecisionPlan(
            disposition="answer",
            decision="Should we wait 3 days?",
            action="Hold the release.",
            reason="We have 3 days before cutoff.",
            supports=[Support("D004", "Hold the release.", "primary")],
            facts=["3 days before cutoff"],
        )
        self.assertTrue(accept_edit(valid_plan, "Because you have 3 days before cutoff, waiting is supported."))

    def test_qualification_retention_preserved(self):
        """Limits and caveats must not be stripped or negated by the editor."""
        self.assertFalse(accept_edit(self.plan, "You should hear both parties without limit or qualification."))
        self.assertFalse(accept_edit(self.plan, "Proceed regardless of any misconduct."))

        valid_explanation = "Hearing both accounts allows the team to understand the dispute before deciding."
        self.assertTrue(accept_edit(self.plan, valid_explanation))

        # Test rendering: if qualification not included in explanation, render_plan appends it
        self.plan.applied = valid_explanation
        rendered = render_plan(self.plan)
        self.assertIn("This would change if verified misconduct requires a report.", rendered)

    def test_second_reader_end_to_end_accepted(self):
        """Structured Gemini mock response that passes all checks gets accepted."""
        mock_response = {
            "verdict": "supported",
            "claims": [
                {
                    "claim": "Disputes require hearing both parties first.",
                    "passage_id": "D004",
                    "evidence_span": "hear both parties in the dispute",
                }
            ],
            "missing_fact": None,
            "suggested_explanation": "Arthashastra 1.15.5 teaches that hearing both accounts before deciding establishes fairness.",
        }

        mock_http = MagicMock()
        mock_http.status_code = 200
        mock_http.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(mock_response)}]}}
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 45,
                "totalTokenCount": 165,
            },
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_EDITOR": "1"}):
            with patch("httpx.post", return_value=mock_http):
                res = review_grounded_reply(self.plan, MOCK_RECORDS)
                self.assertIsNotNone(res)
                self.assertTrue(res["accepted"])
                self.assertEqual(res["verdict"], "supported")
                self.assertEqual(res["explanation"], mock_response["suggested_explanation"])
                self.assertEqual(len(res["claims"]), 1)
                self.assertEqual(res["token_usage"]["total_tokens"], 165)

    def test_second_reader_unsupported_or_clarification(self):
        """Needs clarification routes to missing fact and does not accept the edit."""
        mock_response = {
            "verdict": "needs_clarification",
            "claims": [
                {
                    "claim": "Hearing parties depends on whether misconduct is alleged.",
                    "passage_id": "D004",
                    "evidence_span": "verified misconduct requires a report",
                }
            ],
            "missing_fact": "anyone has alleged criminal fraud or misconduct",
            "suggested_explanation": None,
        }

        mock_http = MagicMock()
        mock_http.status_code = 200
        mock_http.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps(mock_response)}]}}
            ],
        }

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key", "GEMINI_EDITOR": "1"}):
            with patch("httpx.post", return_value=mock_http):
                res = review_grounded_reply(self.plan, MOCK_RECORDS)
                self.assertIsNotNone(res)
                self.assertFalse(res["accepted"])
                self.assertEqual(res["verdict"], "needs_clarification")
                self.assertEqual(res["missing_fact"], "anyone has alleged criminal fraud or misconduct")


if __name__ == "__main__":
    unittest.main()

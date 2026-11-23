"""Run from repository root with python -m unittest discover -s scripts -p test_beta_chat.py."""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app.generator import detect_query_intent, formulate_retrieval_query
from app.server import app, SESSIONS


class BetaChat(unittest.TestCase):
    def test_short_turns(self):
        h = [{"role": "user", "content": "We are releasing a feature."},
             {"role": "assistant", "metadata": {"intent": "new_query"}}]
        for text in ["Help me decide", "Should I change jobs?", "I know my options"]:
            self.assertEqual(detect_query_intent(text, h), "new_query")
        follow = "What if the deadline moves to today?"
        self.assertEqual(detect_query_intent(follow, h), "followup")
        query = formulate_retrieval_query(follow, h, "followup")
        self.assertIn("software feature", query)
        self.assertIn("today", query)
        self.assertEqual(formulate_retrieval_query("release unused funds", h, "new_query"), "release unused funds")

    def test_sse_fallback_and_local_provider(self):
        class Engine:
            def predict(self, query):
                return {"top3": [], "abstained": False, "primary": None}
        app.state.engine = Engine()
        app.state.records = {}
        app.state.model_lock = asyncio.Lock()
        client = TestClient(app)
        def send(msg):
            r = client.post("/api/chat", json={"message": msg, "session_id": "beta-test"})
            self.assertEqual(r.status_code, 200)
            return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]
        try:
            with patch("app.server.local_model_available", AsyncMock(return_value=False)):
                events = send("Help me decide on a launch date")
                self.assertEqual(events[0]["intent"], "new_query")
                self.assertIn("sources", events[1]["text"])
            async def stream(prompt, query):
                self.assertIn("launch date", query)
                yield "Local reply"
            with patch("app.server.local_model_available", AsyncMock(return_value=True)), patch("app.server.stream_answer", stream):
                events = send("What if the deadline changes?")
                self.assertEqual(events[1]["text"], "Local reply")
                self.assertEqual(events[-1]["status"], "complete")
        finally:
            SESSIONS.pop("beta-test", None)
            client.close()

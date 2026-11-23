"""Demonstrate three distinct conversations with Dharm:
1. Everyday Decision: Side-by-side comparison with Gemini-OFF vs. Gemini-ON.
2. Constraint Correction: Multi-turn dilemma with updated facts and deadlines.
3. Clarification & Boundary Abstention: Out-of-scope stock tip abstention followed by bare choice clarification.
"""
import asyncio
import json
import os
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
ML_DIR = BASE_DIR / "decision-corpus-ml"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from fastapi.testclient import TestClient
from app.server import app, SESSIONS, _load_local_env
from rc1_inference import DharmRC1Engine

_load_local_env()


def parse_sse_events(response_text: str):
    events = []
    current_event = None
    for line in response_text.splitlines():
        if line.startswith("event: "):
            current_event = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
    return events


def run_demos():
    print("Initializing engine for demonstration...")
    engine = DharmRC1Engine(device="cpu")
    app.state.engine = engine
    app.state.records = engine.records
    app.state.model_lock = asyncio.Lock()
    client = TestClient(app)

    print("\n" + "=" * 75)
    print("CONVERSATION 1: EVERYDAY DECISION (GEMINI-OFF VS. GEMINI-ON)")
    print("=" * 75)

    msg1 = "My manager asked me to take lead credit on a presentation that an intern spent weeks building. The promotion committee meets on Friday. Do I accept the sole credit or insist on co-presenting?"
    print(f"\nUser: {msg1}\n")

    token = os.getenv("DHARM_BACKEND_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Turn A: Gemini OFF
    os.environ["GEMINI_EDITOR"] = "0"
    resp_off = client.post("/api/chat", headers=headers, json={"message": msg1, "session_id": "demo-c1-off"})
    events_off = parse_sse_events(resp_off.text)
    meta_off = [e["data"] for e in events_off if e["event"] == "metadata"][0]
    chunk_off = [e["data"]["text"] for e in events_off if e["event"] == "chunk"][0]

    print("--- [Mode: Gemini-OFF (Deterministic Dharm Template)] ---")
    print(f"Provider Mode : {meta_off['provider_mode']}")
    print(f"Disposition   : {meta_off['disposition']}")
    print(f"Cited Sources : {[c['work'] + ' ' + c['locator'] for c in meta_off['citations']]}")
    print(f"\nReply:\n{chunk_off}\n")

    # Turn B: Gemini ON
    os.environ["GEMINI_EDITOR"] = "1"
    resp_on = client.post("/api/chat", headers=headers, json={"message": msg1, "session_id": "demo-c1-on"})
    events_on = parse_sse_events(resp_on.text)
    meta_on = [e["data"] for e in events_on if e["event"] == "metadata"][0]
    chunk_on = [e["data"]["text"] for e in events_on if e["event"] == "chunk"][0]

    print("--- [Mode: Gemini-ON (Grounded Second Reader)] ---")
    print(f"Provider Mode : {meta_on['provider_mode']}")
    print(f"Disposition   : {meta_on['disposition']}")
    print(f"Editor Verdict: {meta_on.get('editor_verdict')}")
    print(f"Cited Sources : {[c['work'] + ' ' + c['locator'] for c in meta_on['citations']]}")
    print(f"\nReply:\n{chunk_on}\n")

    print("\n" + "=" * 75)
    print("CONVERSATION 2: MULTI-TURN CONSTRAINT CORRECTION")
    print("=" * 75)

    os.environ["GEMINI_EDITOR"] = "0"
    session_c2 = "demo-c2-correction"

    # Turn 1
    t1_msg = "My team disagrees about releasing a new database migration. We have a week before the freeze. What should we do first?"
    print(f"\n[Turn 1] User: {t1_msg}\n")
    resp_t1 = client.post("/api/chat", headers=headers, json={"message": t1_msg, "session_id": session_c2})
    events_t1 = parse_sse_events(resp_t1.text)
    meta_t1 = [e["data"] for e in events_t1 if e["event"] == "metadata"][0]
    chunk_t1 = [e["data"]["text"] for e in events_t1 if e["event"] == "chunk"][0]
    print(f"Assistant (Intent: {meta_t1['intent']}, Disp: {meta_t1['disposition']}):")
    print(f"{chunk_t1}\n")

    # Turn 2: Fact update
    t2_msg = "Actually, the freeze deadline moved to this afternoon, but the rollback window stays open."
    print(f"[Turn 2] User (Correction): {t2_msg}\n")
    resp_t2 = client.post("/api/chat", headers=headers, json={"message": t2_msg, "session_id": session_c2})
    events_t2 = parse_sse_events(resp_t2.text)
    meta_t2 = [e["data"] for e in events_t2 if e["event"] == "metadata"][0]
    chunk_t2 = [e["data"]["text"] for e in events_t2 if e["event"] == "chunk"][0]
    print(f"Assistant (Intent: {meta_t2['intent']}, Disp: {meta_t2['disposition']}):")
    print(f"Updated Facts in Plan: {meta_t2['plan']['facts']}")
    print(f"\nReply:\n{chunk_t2}\n")

    print("\n" + "=" * 75)
    print("CONVERSATION 3: CLARIFICATION & BOUNDARY ABSTENTION")
    print("=" * 75)

    session_c3 = "demo-c3-boundary"

    # Turn 1: Out of scope financial tip
    t1_bound = "Which stocks should I buy tomorrow morning with $10,000 to maximize short-term capital gains?"
    print(f"\n[Turn 1] User (Illegal/Financial Tip): {t1_bound}\n")
    resp_b1 = client.post("/api/chat", headers=headers, json={"message": t1_bound, "session_id": session_c3})
    events_b1 = parse_sse_events(resp_b1.text)
    meta_b1 = [e["data"] for e in events_b1 if e["event"] == "metadata"][0]
    chunk_b1 = [e["data"]["text"] for e in events_b1 if e["event"] == "chunk"][0]
    print(f"Assistant (Intent: {meta_b1['intent']}, Disp: {meta_b1['disposition']}):")
    print(f"{chunk_b1}\n")

    # Turn 2: Bare choice
    t2_bare = "Should I work out in the morning or evening?"
    print(f"[Turn 2] User (Bare Choice): {t2_bare}\n")
    resp_b2 = client.post("/api/chat", headers=headers, json={"message": t2_bare, "session_id": session_c3})
    events_b2 = parse_sse_events(resp_b2.text)
    meta_b2 = [e["data"] for e in events_b2 if e["event"] == "metadata"][0]
    chunk_b2 = [e["data"]["text"] for e in events_b2 if e["event"] == "chunk"][0]
    print(f"Assistant (Intent: {meta_b2['intent']}, Disp: {meta_b2['disposition']}):")
    print(f"{chunk_b2}\n")

    print("=" * 75)
    print("DEMONSTRATION RUN COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_demos()

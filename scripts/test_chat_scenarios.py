"""Test all 4 conversational scenarios against the running Dharm FastAPI server.

Scenarios tested:
1. Fresh dilemma (Three villages ferry landing)
2. Constraint-changing follow-up ("What if the deadline moves to today?")
3. Unrelated new topic ("How do I fix a leaking kitchen pipe?") -> tests epistemic abstention
4. Ordinary chat / greeting ("Hello, what can you assist me with?")
"""
import json
import urllib.request

SERVER_URL = "http://127.0.0.1:8000/api/chat"


def query_chat_stream(message: str, session_id: str):
    payload = json.dumps({"message": message, "session_id": session_id}).encode("utf8")
    req = urllib.request.Request(
        SERVER_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    metadata = None
    chunks = []

    with urllib.request.urlopen(req, timeout=60.0) as resp:
        buffer = ""
        for raw_line in resp:
            line = raw_line.decode("utf8")
            buffer += line
            if buffer.endswith("\n\n"):
                blocks = buffer.strip().split("\n\n")
                buffer = ""
                for block in blocks:
                    event_type = "chunk"
                    data_str = ""
                    for subline in block.split("\n"):
                        if subline.startswith("event: "):
                            event_type = subline[7:].strip()
                        elif subline.startswith("data: "):
                            data_str = subline[6:].strip()

                    if event_type == "metadata":
                        metadata = json.loads(data_str)
                    elif event_type == "chunk":
                        chunk_obj = json.loads(data_str)
                        chunks.append(chunk_obj["text"])

    return metadata, "".join(chunks)


def main():
    print("=" * 80)
    print("Exercising Dharm Live Chat Conversational Scenarios")
    print("=" * 80)

    session_1 = "test_session_villages"

    # Scenario 1: Fresh Dilemma
    print("\n--- Scenario 1: Fresh Dilemma ---")
    q1 = "Three villages share a ferry landing. Their representatives meet Friday to agree a temporary access plan. A clerk can reserve the only accessible berth without deciding the plan. The reservation expires next month. What comes first?"
    print(f"User: {q1}")
    meta1, text1 = query_chat_stream(q1, session_1)
    print(f"Assistant Primary: {meta1['primary']}")
    print(f"Assistant Top-3:   {meta1['top3']}")
    print(f"Abstained:        {meta1['abstained']}")
    print(f"Citations:        {[c['id'] for c in meta1['citations']]}")
    print(f"Streamed Text:\n{text1[:300]}...\n")

    # Scenario 2: Constraint-changing Follow-up
    print("--- Scenario 2: Constraint-changing Follow-up ---")
    q2 = "What if the deadline moves to today?"
    print(f"User: {q2}")
    meta2, text2 = query_chat_stream(q2, session_1)
    print(f"Reformulated:     {meta2.get('reformulated_query')}")
    print(f"Assistant Primary: {meta2['primary']}")
    print(f"Assistant Top-3:   {meta2['top3']}")
    print(f"Abstained:        {meta2['abstained']}")
    print(f"Citations:        {[c['id'] for c in meta2['citations']]}")
    print(f"Streamed Text:\n{text2[:300]}...\n")

    # Scenario 3: Unrelated New Topic (in a fresh session to test isolation)
    session_2 = "test_session_plumbing"
    print("--- Scenario 3: Unrelated Out-of-Scope Topic ---")
    q3 = "How do I fix a leaking kitchen pipe?"
    print(f"User: {q3}")
    meta3, text3 = query_chat_stream(q3, session_2)
    print(f"Assistant Primary: {meta3['primary']}")
    print(f"Abstained:        {meta3['abstained']} (prob = {meta3['abstention_prob']:.4f})")
    print(f"Streamed Text:\n{text3[:300]}...\n")

    # Scenario 4: Ordinary Chat / Greeting
    session_3 = "test_session_greeting"
    print("--- Scenario 4: Ordinary Chat / Greeting ---")
    q4 = "Hello, what can you assist me with?"
    print(f"User: {q4}")
    meta4, text4 = query_chat_stream(q4, session_3)
    print(f"Intent:           {meta4['intent']}")
    print(f"Streamed Text:\n{text4}\n")


if __name__ == "__main__":
    main()

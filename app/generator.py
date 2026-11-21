"""Server-Side Answer Generation Adapter for Dharm.

Supports:
1. Live streaming via Google Gemini API (if GEMINI_API_KEY or GOOGLE_API_KEY is configured).
2. Local Retrieval Preview Mode (if no API key is configured), generating a structured
   synthesis preview from the retrieved passages, derived principles, sequence roles,
   and modern applications, clearly labeled as preview mode.

Handles conversational behavior:
- New dilemma: identifies core conflict, gives direct practical answer first,
  distinguishes text's root teaching from modern institutional application.
- Follow-up query: carries forward scenario facts, updates constraints, and reruns retrieval.
- Greetings / ordinary chat: natural, conversational response without hallucinating corpus rulings.
- Out-of-scope / ungroundable topics: epistemic abstention explaining what the corpus
  can establish and where its guidance ends.
"""
import json
import os
import re
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

GEMINI_MODEL = "gemini-2.5-flash"
API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:streamGenerateContent"


def get_api_key() -> Optional[str]:
    """Retrieve Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def detect_query_intent(text: str, history: List[Dict[str, str]]) -> str:
    """Classify user turn intent into: 'greeting', 'followup', 'out_of_scope', or 'new_query'."""
    cleaned = text.strip().lower()

    # Greetings & meta-questions
    greeting_exact = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "namaste", "help", "who are you", "what can you do", "what do you do"
    }
    if cleaned in greeting_exact:
        return "greeting"

    greeting_prefixes = ("hi ", "hello ", "hey ", "namaste ", "help ", "greetings")
    if any(cleaned.startswith(p) for p in greeting_prefixes) and len(cleaned.split()) <= 10:
        return "greeting"

    greeting_phrases = ["what can you assist", "what can you help", "how can you help", "what is dharm"]
    if any(phrase in cleaned for phrase in greeting_phrases):
        return "greeting"

    # Out-of-scope non-governance queries (physical repairs, coding, cooking, medical)
    out_of_scope_patterns = [
        "leaking kitchen pipe", "plumbing", "fix a pipe", "recipe for", "bake a cake",
        "write a python script to", "css flexbox", "treat a fever", "car engine oil"
    ]
    if any(pat in cleaned for pat in out_of_scope_patterns):
        return "out_of_scope"

    # Follow-up indicators (if history exists)
    if history and len(history) >= 2:
        followup_signals = [
            "what if", "suppose", "instead", "now", "moves to", "changes to", "how about",
            "and then", "what about", "in that case", "if the deadline", "what should we do then",
            "can we still", "why", "which one", "explain"
        ]
        if any(sig in cleaned for sig in followup_signals) or len(cleaned.split()) <= 8:
            return "followup"

    return "new_query"


def formulate_retrieval_query(current_message: str, history: List[Dict[str, str]], intent: str) -> str:
    """Formulate the dilemma text to pass to DharmRC1Engine."""
    if intent == "followup" and history:
        # Find the most recent substantive user dilemma in history
        last_dilemma = ""
        for turn in reversed(history):
            if turn.get("role") == "user" and len(turn.get("content", "")) > 30:
                last_dilemma = turn["content"]
                break

        if last_dilemma:
            # Combine background scenario with modified constraint
            return f"Background Context: {last_dilemma}. Updated Situation / Changed Constraint: {current_message}"

    return current_message


def build_system_prompt(records: Dict[str, Dict[str, Any]],
                        retrieval_res: Dict[str, Any],
                        intent: str) -> str:
    """Construct the grounding system prompt for generation."""
    if intent == "greeting":
        return (
            "You are Dharm, an institutional decision assistant grounded in classical governance "
            "and legal principles (including the Arthashastra and Chanakya Niti). Respond naturally, "
            "welcoming the user, and explain that you help institutional leaders, committees, and administrators "
            "navigate complex governance impasses, priority conflicts, and sequential decision workflows."
        )

    if retrieval_res.get("abstained"):
        return (
            "You are Dharm, an institutional decision assistant grounded in classical governance texts. "
            "The epistemic abstention gate determined that this question lies OUTSIDE the corpus's domain of "
            "governance, political ethics, and institutional arbitration. "
            "State clearly and gracefully what the Dharm corpus can establish, and where its guidance ends. "
            "Explain that the corpus does not provide authoritative guidance on this specific matter, and decline "
            "to make up principles."
        )

    # Build passage grounding text
    top3_ids = retrieval_res.get("top3", [])
    primary_id = retrieval_res.get("primary")
    promoted_pair = retrieval_res.get("promoted_pair")

    passages_text = []
    for pid in top3_ids:
        if pid in records:
            r = records[pid]
            role = r.get("complementary_passage_id", {}).get("sequence_role", "Independent")
            comp = r.get("competing_passage_id", {}).get("condition", "None")
            scen = r["scenarios_json"][0]["scenario"] if r["scenarios_json"] else ""
            passages_text.append(
                f"- Passage [{pid}] ({r['work']} {r['locator']}):\n"
                f"  Principle: {r['derived_principle']}\n"
                f"  Text: {r['english_translation']}\n"
                f"  Context Summary: {r['context_summary']}\n"
                f"  Sequence Role: {role}\n"
                f"  Competing Contrast Condition: {comp}\n"
                f"  Modern Institutional Application: {scen}\n"
            )

    passages_block = "\n".join(passages_text)

    is_synthesis = bool(promoted_pair and len(promoted_pair) == 2)
    workflow_note = ""
    if is_synthesis:
        workflow_note = (
            f"NOTE: The verifier has promoted an ordered two-step workflow: first apply [{promoted_pair[0]}], "
            f"then transition to [{promoted_pair[1]}]. Detail the sequential execution order."
        )
    else:
        workflow_note = f"Primary Governing Decision: Passage [{primary_id}]."

    return (
        "You are Dharm, an authoritative decision advisor for institutional leaders and administrators, "
        "grounded in classical Indian statecraft (Arthashastra, Chanakya traditions). "
        "Your task is to resolve the user's dilemma by applying the retrieved governing passages.\n\n"
        "Grounding Rules:\n"
        "1. Direct Practical Ruling First: Give an immediate, decisive, actionable answer in the opening sentence.\n"
        "2. Ancient Teaching vs Modern Application: Clearly distinguish the historical root principle "
        "from its modern organizational implementation.\n"
        "3. Sequential Logic: If a two-step workflow was promoted, clearly explain step 1 and step 2 in order.\n"
        "4. Application Limits & Boundary Conditions: Explicitly state under what circumstances this advice holds, "
        "and when it would change (e.g. if the contrast condition arises).\n"
        "5. Citations: Cite passages inline using brackets like [D019] or [D042] whenever referencing their principles.\n"
        f"\n{workflow_note}\n\n"
        f"Retrieved Authoritative Passages:\n{passages_block}"
    )


async def stream_gemini_response(api_key: str,
                                 system_prompt: str,
                                 user_message: str,
                                 history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Stream answer chunks from Gemini API via Server-Sent Events."""
    url = f"{API_ENDPOINT}?alt=sse&key={api_key}"

    # Build conversation contents
    contents = []
    # Include up to last 4 turns for context
    recent_history = history[-4:] if len(history) > 4 else history
    for turn in recent_history:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": turn.get("content", "")}]
        })

    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        }
    }

    req_data = json.dumps(payload).encode("utf8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            for line_bytes in resp:
                line = line_bytes.decode("utf8").strip()
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_obj = json.loads(data_str)
                        candidates = chunk_obj.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text_piece = part.get("text", "")
                                if text_piece:
                                    yield text_piece
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        yield f"\n\n*[API Streaming Error: {str(e)}]*"


async def stream_retrieval_preview_response(records: Dict[str, Dict[str, Any]],
                                            retrieval_res: Dict[str, Any],
                                            user_message: str,
                                            intent: str) -> AsyncGenerator[str, None]:
    """Generate structured response in Retrieval Preview Mode when no API key is set."""
    yield "### [Retrieval Preview Mode: Live Generation Pending GEMINI_API_KEY]\n\n"

    if intent == "greeting":
        yield (
            "Welcome to **Dharm**. I am your institutional decision assistant grounded in classical "
            "governance texts (Arthashastra, Chanakya Niti). I assist administrators and committees in navigating "
            "competing responsibilities, sequential workflows, and ethical impasses.\n\n"
            "Present a practical dilemma or institutional conflict to begin."
        )
        return

    if intent == "out_of_scope" or retrieval_res.get("abstained"):
        prob = retrieval_res.get("abstention_prob", 1.0)
        yield (
            f"**Status: Epistemic Abstention** (Abstention Probability: `{prob:.4f}`)\n\n"
            "The Dharm decision corpus covers **political statecraft, public administration, and institutional arbitration**.\n\n"
            "This query lies outside the corpus's domain of institutional governance. "
            "The system deliberately declines to produce an ungrounded recommendation where classical statecraft principles "
            "cannot provide rigorous backing. It can establish authoritative guidance on organizational priority conflicts, "
            "sequential workflows, and administrative authority, but does not extend to general physical or domestic maintenance."
        )
        return

    primary_id = retrieval_res.get("primary")
    top3_ids = retrieval_res.get("top3", [])
    promoted_pair = retrieval_res.get("promoted_pair")
    pair_promoted = retrieval_res.get("pair_promoted", False)

    primary_rec = records.get(primary_id, {})
    prin = primary_rec.get("derived_principle", "")
    trans = primary_rec.get("english_translation", "")
    scen = primary_rec.get("scenarios_json", [{}])[0].get("scenario", "")
    contrast = primary_rec.get("competing_passage_id", {})

    yield f"**Governing Decision**: Apply [{primary_id}] ({primary_rec.get('work')} {primary_rec.get('locator')})\n\n"

    if pair_promoted and promoted_pair:
        u_id, v_id = promoted_pair
        u_rec = records.get(u_id, {})
        v_rec = records.get(v_id, {})
        yield (
            f"**Sequential Workflow (Two-Phase Strategy)**:\n"
            f"1. **Phase 1 — [{u_id}]**: {u_rec.get('derived_principle')}\n"
            f"2. **Phase 2 — [{v_id}]**: {v_rec.get('derived_principle')}\n"
            f"   *Transition Rule*: {u_rec.get('complementary_passage_id', {}).get('sequence_role', 'Follow with')}\n\n"
        )
    else:
        yield f"**Primary Action**: {prin}\n\n"

    yield (
        "**Root Teaching vs Modern Application**:\n"
        f"- *Classical Teaching*: \"{trans}\"\n"
        f"- *Modern Institutional Context*: {scen}\n\n"
        "**Boundary Conditions & Application Limits**:\n"
        f"- *Standard Rule*: Follow [{primary_id}] while circumstances remain standard.\n"
        f"- *Contrast Condition*: If *{contrast.get('condition', 'competing constraints arise')}*, "
        f"arbitration shifts priority to [{contrast.get('id')}]: \"{contrast.get('flip_to_competing_when', '')}\".\n\n"
        "*(Set `GEMINI_API_KEY` on the server to enable fully conversational synthesized responses.)*"
    )

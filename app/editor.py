"""Source-grounded Gemini second reader & copy editor.

Dharm deterministically owns the chosen disposition, action, ordered second step,
application limit, and source citations.

Gemini serves strictly as an opt-in second reader:
- Reviews support against a compact evidence bundle (classical text separated from modern interpretation).
- Enforces strict server-side validation: evidence spans must occur verbatim in the supplied corpus text.
- Rejects reversed actions, invented facts, qualification stripping, or unapproved passage IDs.
- Consolidates review, claim extraction, and explanation into a SINGLE paid network call.
- Any timeout, quota error, invalid JSON, or disagreement falls back safely to Dharm's local response.
"""
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

_ID = re.compile(r"\b([A-Za-z]{1,4}\d{2,4})\b")
_NOT = re.compile(r"\b(do not|don't|never|avoid|instead of|rather than|the opposite)\b", re.I)

# Approved contrasts. A paraphrase can reverse an action without reusing its verb.
_CONTRARY = (
    (("hold", "wait", "delay", "postpone", "until"),
     ("immediately", "right away", "ship now", "shipping immediately", "release now", "ship it", "proceed now")),
    (("invite", "describe", "hear", "consult", "convene"),
     ("skip the conversation", "don't ask", "do not ask", "ignore them", "unilaterally decide", "bypass discussion")),
    (("disclose", "declare", "report", "admit", "speak up"),
     ("stay quiet", "conceal", "hide it", "keep quiet", "cover up", "ignore")),
    (("preserve", "reserve", "protect", "save"),
     ("spend it all", "exhaust", "deplete", "consume now")),
)

REVIEW_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "STRING",
            "enum": ["supported", "needs_clarification", "unsupported"],
        },
        "claims": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "passage_id": {"type": "STRING"},
                    "evidence_span": {"type": "STRING"},
                },
                "required": ["claim", "passage_id", "evidence_span"],
            },
        },
        "missing_fact": {"type": "STRING", "nullable": True},
        "suggested_explanation": {"type": "STRING", "nullable": True},
    },
    "required": ["verdict", "claims"],
}


def editor_enabled() -> bool:
    """Strictly opt-in: enabled only when GEMINI_EDITOR is 1/true and GEMINI_API_KEY is present."""
    val = os.getenv("GEMINI_EDITOR", "0").strip().lower()
    if val not in ("1", "true", "yes", "on"):
        return False
    return bool(os.getenv("GEMINI_API_KEY"))


def build_evidence_bundle(plan, records: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Construct compact evidence bundle for cited steps only.

    Labels classical translation and modern editorial interpretation separately.
    Sources merely retrieved in top-3 but not cited are excluded.
    """
    bundle = []
    for step in plan.supports:
        pid = step.id.upper()
        if pid in records:
            r = records[pid]
            bundle.append({
                "passage_id": pid,
                "role": step.role,
                "work": r.get("work", ""),
                "locator": r.get("locator", ""),
                "classical_text": r.get("english_translation", ""),
                "modern_editorial_interpretation": r.get("derived_principle", ""),
                "application_limit": r.get("application_limit", ""),
                "relevant_user_facts": list(plan.facts or []),
            })
    return bundle


def verify_evidence_spans(claims: List[Dict[str, Any]], plan, records: Dict[str, Dict[str, Any]]) -> bool:
    """Enforce that every evidence span occurs verbatim in that passage's text in the corpus."""
    allowed_ids = {s.id.upper() for s in plan.supports}
    if not claims:
        return False
    for claim in claims:
        pid = claim.get("passage_id", "").upper()
        if pid not in allowed_ids or pid not in records:
            return False
        span = claim.get("evidence_span", "").strip()
        if not span or len(span) < 3:
            return False
        rec = records[pid]
        supplied = (
            (rec.get("english_translation") or "") + " " +
            (rec.get("derived_principle") or "") + " " +
            (rec.get("application_limit") or "")
        ).casefold()
        if span.casefold() not in supplied:
            return False
    return True


def _contradicts(action: str, text: str) -> bool:
    folded_action = (action or "").casefold()
    folded = text.casefold()
    for ours, opposite in _CONTRARY:
        if any(token in folded_action for token in ours) and any(token in folded for token in opposite):
            return True
        if any(token in folded_action for token in opposite) and any(token in folded for token in ours):
            return True
    return False


def _invented(plan, text: str, records: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    source_parts = [
        plan.decision, plan.action, plan.second_action or "", plan.reason, plan.limit or "",
        getattr(plan, "teaching", "") or "",
        " ".join(plan.facts or []),
    ]
    if records:
        for s in plan.supports:
            pid = s.id.upper()
            if pid in records:
                rec = records[pid]
                source_parts.append(rec.get("locator") or "")
                source_parts.append(rec.get("work") or "")
    source = " ".join(source_parts).casefold()
    for number in re.findall(r"\b\d[\d,]*\b", text):
        if number.casefold() not in source:
            return True
    return False


def _verbs(action: str) -> List[str]:
    return re.findall(r"\b[a-z]{4,}\b", (action or "").casefold())[:4]


def accept_edit(plan, explanation: str, records: Optional[Dict[str, Dict[str, Any]]] = None) -> bool:
    """Reject an explanation that adds unapproved sources, reverses action, or invents facts."""
    text = (explanation or "").strip()
    if not text or len(text) > 800:
        return False
    allowed = {item.id.upper() for item in plan.supports}
    mentioned = {match.group(1).upper() for match in _ID.finditer(text)}
    if mentioned - allowed:
        return False
    if mentioned or re.search(r"\b(first|next|then),", text, re.I):
        return False
    folded = text.casefold()
    if _contradicts(plan.action, text):
        return False
    if _invented(plan, text, records):
        return False
    if not _NOT.search(plan.action or ""):
        for verb in _verbs(plan.action):
            if re.search(
                rf"\b(?:do not|don't|never|avoid|instead of|rather than|the opposite)\s+(?:\w+\s+){{0,3}}{re.escape(verb)}\b",
                folded,
            ):
                return False
    # Verify qualification retention: must not contradict plan.limit if present
    if plan.limit:
        if "without limit" in folded or "regardless of" in folded:
            return False
    return True


def _gemini_call(prompt: str, schema: Dict[str, Any], timeout: float = 12.0) -> Tuple[Optional[Dict[str, Any]], float, Optional[Dict[str, int]]]:
    """Execute single Google Gemini API call with structured schema and timing."""
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None, 0.0, None
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 600,
            "thinkingConfig": {"thinkingBudget": 0},
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    t0 = time.perf_counter()
    try:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except (httpx.HTTPError, httpx.TimeoutException):
        return None, time.perf_counter() - t0, None

    latency = time.perf_counter() - t0
    if response.status_code != 200:
        return None, latency, None

    try:
        import json
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        usage = body.get("usageMetadata")
        tokens = None
        if usage:
            tokens = {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "candidates_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
        return json.loads(text), latency, tokens
    except (KeyError, IndexError, TypeError, ValueError):
        return None, latency, None


def review_grounded_reply(plan, records: Dict[str, Dict[str, Any]], timeout: float = 12.0) -> Optional[Dict[str, Any]]:
    """Single-call source-grounded second reader.

    Returns structured evaluation with verdict, verified claims, explanation, latency, and tokens.
    Returns None if editor is disabled or on any unrecoverable failure.
    """
    if not editor_enabled() or plan.disposition != "answer":
        return None

    bundle = build_evidence_bundle(plan, records)
    if not bundle:
        return None

    import json
    prompt = (
        "You are an expert NLP second reader and evidence verifier for Dharm governance decisions.\n"
        "Your task is to evaluate whether the proposed action follows from the supplied cited texts.\n\n"
        f"User Decision Dilemma: {plan.decision}\n"
        f"User Facts: {json.dumps(plan.facts or [])}\n"
        f"Proposed Action: {plan.action}\n"
        f"Second Action: {plan.second_action or 'none'}\n"
        f"Application Limit: {plan.limit or 'none'}\n"
        f"Evidence Bundle (Cited Steps):\n{json.dumps(bundle, indent=2)}\n\n"
        "Instructions:\n"
        "1. Determine verdict: 'supported' if the action directly derives from the texts, "
        "'needs_clarification' if a critical user fact is missing to settle it, "
        "or 'unsupported' if the texts do not justify this recommendation.\n"
        "2. Extract claims: for each key claim, specify passage_id and the EXACT verbatim evidence_span "
        "from the supplied text.\n"
        "3. If supported, write suggested_explanation (2-4 plain sentences). "
        "Do NOT reverse or negate the proposed action. Do NOT invent new facts or external statistics. "
        "Name the work and locator once.\n"
        "Return structured JSON matching the schema."
    )

    review_json, latency, tokens = _gemini_call(prompt, REVIEW_SCHEMA, timeout=timeout)
    if not review_json or "verdict" not in review_json or "claims" not in review_json:
        return None

    # Server-side verification
    claims = review_json.get("claims", [])
    if not verify_evidence_spans(claims, plan, records):
        # Spans do not occur in corpus -> invented evidence! Fallback immediately.
        return {
            "verdict": "rejected_invented_spans",
            "accepted": False,
            "explanation": None,
            "latency_sec": latency,
            "token_usage": tokens,
        }

    verdict = review_json.get("verdict")
    missing_fact = review_json.get("missing_fact")
    suggested = (review_json.get("suggested_explanation") or "").strip()

    if verdict == "supported":
        if accept_edit(plan, suggested, records):
            return {
                "verdict": "supported",
                "accepted": True,
                "explanation": suggested,
                "claims": claims,
                "latency_sec": latency,
                "token_usage": tokens,
            }
        else:
            return {
                "verdict": "supported",
                "accepted": False,
                "reason": "explanation_validation_failed",
                "explanation": None,
                "claims": claims,
                "latency_sec": latency,
                "token_usage": tokens,
            }

    return {
        "verdict": verdict,
        "accepted": False,
        "missing_fact": missing_fact,
        "claims": claims,
        "latency_sec": latency,
        "token_usage": tokens,
    }


def synthesize_grounded_reply(plan, evidence: Any = None, records: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[str]:
    """Compatibility wrapper around review_grounded_reply returning explanation."""
    if records is None:
        return None
    res = review_grounded_reply(plan, records)
    if res and res.get("accepted"):
        return res.get("explanation")
    return None

"""Optional Gemini copy editor.

Dharm keeps the disposition, action, second action, limit, and source IDs.
Gemini may rewrite only the explanatory reason. One call is made per eligible
answer. Any timeout, missing key, rate limit, malformed payload, invented
source, or reversed action returns the original reason.
"""
import os
import re

import httpx

_ID = re.compile(r"\b([A-Za-z]{1,4}\d{2,4})\b")
_NOT = re.compile(r"\b(do not|don't|never|avoid|instead of|rather than|the opposite)\b", re.I)
# Approved contrasts. A paraphrase can reverse an action without reusing its verb.
_CONTRARY = (
    (("hold", "wait", "delay", "postpone", "until"),
     ("immediately", "right away", "ship now", "shipping immediately", "release now", "ship it")),
    (("invite", "describe", "hear"),
     ("skip the conversation", "don't ask", "do not ask", "ignore them")),
)


def editor_enabled():
    """On when an API key is present, unless GEMINI_EDITOR=0."""
    if os.getenv("GEMINI_EDITOR", "1") == "0":
        return False
    return bool(os.getenv("GEMINI_API_KEY"))


def _contradicts(action, text):
    folded_action = (action or "").casefold()
    folded = text.casefold()
    for ours, opposite in _CONTRARY:
        if any(token in folded_action for token in ours) and any(token in folded for token in opposite):
            return True
        if any(token in folded_action for token in opposite) and any(token in folded for token in ours):
            return True
    return False


def _invented(plan, text):
    source = " ".join([
        plan.decision, plan.action, plan.second_action or "", plan.reason, plan.limit or "",
        getattr(plan, "teaching", "") or "",
        " ".join(plan.facts or []),
    ]).casefold()
    for number in re.findall(r"\b\d[\d,]*\b", text):
        if number.casefold() not in source:
            return True
    return False


def _verbs(action):
    return re.findall(r"\b[a-z]{4,}\b", (action or "").casefold())[:4]


def accept_edit(plan, reason):
    """Reject an edit that adds a source or reverses the approved action."""
    text = (reason or "").strip()
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
    if _invented(plan, text):
        return False
    if not _NOT.search(plan.action or ""):
        for verb in _verbs(plan.action):
            if re.search(
                rf"\b(?:do not|don't|never|avoid|instead of|rather than|the opposite)\s+(?:\w+\s+){{0,3}}{re.escape(verb)}\b",
                folded,
            ):
                return False
    return True


def _gemini_json(prompt, schema, timeout=12.0):
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingBudget": 0},
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        response = httpx.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        import json
        body = response.json()
        text = body["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def action_fits_decision(plan):
    """False only when Gemini says the retrieved action is about a different problem.

    Missing key, errors, or unknown results keep the plan (True).
    """
    if plan.disposition != "answer" or not plan.action:
        return True
    if not os.getenv("GEMINI_API_KEY"):
        return True
    payload = _gemini_json(
        (
            "A retrieval system proposed an ACTION for a user's DECISION. "
            "Say whether that action is actually about this decision. "
            "fits must be false when the action belongs to a different kind of problem "
            "(for example attestation/publication for a job-and-move choice, "
            "or staffing/workload experiments for family caregiving). "
            "Metaphorical stretch is not enough. Return JSON with fits.\n\n"
            f"Decision: {plan.decision}\n"
            f"Action: {plan.action}\n"
            f"Limit: {plan.limit or 'none'}"
        ),
        {
            "type": "OBJECT",
            "properties": {"fits": {"type": "BOOLEAN"}},
            "required": ["fits"],
        },
        timeout=8.0,
    )
    if not payload or "fits" not in payload:
        return True
    return bool(payload["fits"])


def synthesize_grounded_reply(plan, evidence):
    """Apply the retrieved teaching to this decision. None keeps the template."""
    if not os.getenv("GEMINI_API_KEY", "") or plan.disposition != "answer":
        return None
    payload = _gemini_json(
        (
            "Apply the cited teaching to the user's decision in two to four plain sentences. "
            "Name the work and the locator once. Say how the teaching bears on this decision and on any deadline. "
            "Keep the same direction as the action. Do not recommend the opposite. "
            "Do not add a passage id or a number that was not already given. "
            "Return JSON with one field, reply.\n\n"
            f"Decision: {plan.decision}\n"
            f"Teaching: {getattr(plan, 'teaching', '') or 'none'}\n"
            f"Action to preserve: {plan.action}\n"
            f"Second action to preserve: {plan.second_action or 'none'}\n"
            f"Limit to preserve: {plan.limit or 'none'}\n"
            f"Facts: {'; '.join(plan.facts) or 'none'}\n"
            f"Evidence: {evidence}"
        ),
        {
            "type": "OBJECT",
            "properties": {"reply": {"type": "STRING"}},
            "required": ["reply"],
        },
    )
    if not payload:
        return None
    reply = (payload.get("reply") or "").strip()
    if not accept_edit(plan, reply):
        return None
    return reply


def edit_reason(plan, evidence):
    """One server-side edit. Returns a new reason, or None to keep Dharm's."""
    if not os.getenv("GEMINI_API_KEY", "") or plan.disposition != "answer":
        return None
    payload = _gemini_json(
        (
            "Rewrite only the reason so it is easier to read. Describe the user's situation and any deadline. "
            "Do not restate the action, the second step, or the limit. Do not give a new instruction. "
            "Do not add a source or a passage id. Do not recommend the opposite. "
            "Return JSON with one field, reason.\n\n"
            f"Decision: {plan.decision}\n"
            f"Action to preserve: {plan.action}\n"
            f"Second action to preserve: {plan.second_action or 'none'}\n"
            f"Limit to preserve: {plan.limit or 'none'}\n"
            f"Facts: {'; '.join(plan.facts) or 'none'}\n"
            f"Current reason: {plan.reason}\n"
            f"Evidence: {evidence}"
        ),
        {
            "type": "OBJECT",
            "properties": {"reason": {"type": "STRING"}},
            "required": ["reason"],
        },
    )
    if not payload:
        return None
    reason = payload.get("reason")
    if not accept_edit(plan, reason):
        return None
    return reason.strip()

"""Rule-based conversation state for the chat app.

Intent labels, search rewrites, and deadline facts are rules. They do not
choose a passage and they do not invent an answer. Search wording stays out of
the text shown back to the user.
"""
import re

WELCOME_QUESTION = "What decision are you weighing?"
TRADITION = "Dharm draws on the Vedas, Upanishads, Bhagavad Gita, and Chanakya tradition."

_GREETING = re.compile(
    r"(hi|hello|hey|namaste|good (morning|afternoon|evening)|who are you|what can you do|what is dharm)"
)
_HELP_ONLY = re.compile(r"help( me)?( please)?")
_HELP_DECIDE = re.compile(r"help me decide( please)?")
_FOLLOWUP = re.compile(
    r"^(what if|suppose|in that case|instead|what about|can we still|what should we do then|"
    r"why is that|why$|what changes|and if|does that|how soon|the deadline (?:moved|changed)|"
    r"let'?s do|do that|go with that)\b"
)
_CORRECTION = re.compile(
    r"^(actually|no[, ]|wait\b|i meant|correction\b|that(?:'s| is) (?:not|wrong)|it(?:'s| is) not|"
    r"rather[, ]|i misspoke|sorry[, ] i)\b"
)
_OUT_OF_SCOPE = re.compile(
    r"^(?:how (?:do|can) i (?:fix a (?:leaking )?pipe|bake a cake)|write a python script to|give me a recipe for)\b"
)
_CLOSING = re.compile(
    r"\b(this afternoon|today|tonight|this morning|right now|immediately|in (?:an|\d+) hours?)\b",
    re.I,
)
_OPEN = re.compile(r"\b((?:a|one|1) week|next week|this week|in a week)\b", re.I)
_DECISION_WORD = re.compile(r"\b(should|whether|decide|decision|deadline)\b")


def detect_query_intent(text, history):
    words = text.strip().casefold().strip(" .!?")
    if _GREETING.fullmatch(words):
        return "greeting"
    if re.match(r"^(hi|hello|hey|namaste)\b", words) and len(words.split()) <= 12 and not _DECISION_WORD.search(words):
        return "greeting"
    if _HELP_ONLY.fullmatch(words) or _HELP_DECIDE.fullmatch(words):
        return "clarify"
    if re.match(r"^(new topic|different question|unrelated)\b", words):
        return "new_query"
    if _OUT_OF_SCOPE.match(words):
        return "out_of_scope"
    if history and (_FOLLOWUP.match(words) or _CORRECTION.match(words) or _refers_to_prior_reply(words)):
        return "followup"
    # "release it" has no named object. "release the unused budget ... keep it" does.
    if not history and re.search(r"\b(?:release|do|ship|launch|change|decide) it\b", words):
        return "clarify"
    return "new_query"


def _refers_to_prior_reply(words):
    return bool(re.search(
        r"\b(you said|you suggested|your suggestion|the first one|the second one|do that|go with that)\b",
        words,
    ))


def needs_clarification(text, intent):
    """A short follow-up that points at something unnamed."""
    if intent != "followup":
        return False
    if len(text.split()) > 6:
        return False
    folded = text.casefold()
    if _CLOSING.search(text) or _OPEN.search(text) or re.search(r"\d", text):
        return False
    if re.search(r"\b(you said|you suggested|your suggestion|the first|the second)\b", folded):
        return False
    return bool(re.search(r"\b(that|it|this|those|them|the other)\b", folded))


def clarifying_question(message, intent):
    """One question. This is a template, not a retrieved passage."""
    words = message.strip().casefold()
    if intent == "greeting" or _HELP_ONLY.fullmatch(words.strip(" .!?")) or _HELP_DECIDE.fullmatch(words.strip(" .!?")):
        return WELCOME_QUESTION
    if re.search(r"\bit\b", words) and not re.search(r"\bit (?:is|isn't|was|wasn't)\b", words):
        return "What is that decision about?"
    if needs_clarification(message, intent):
        return "Which option do you mean?"
    return WELCOME_QUESTION


def retrieval_wording(text):
    """Search rewrite. The planner and the reply keep the user's original words."""
    return re.sub(
        r"\b(?:releasing|shipping|launching)\s+(?:a |the |our )?(software )?feature\b",
        "rolling out a software feature",
        text,
        flags=re.I,
    )


def topic_pairs(history):
    pairs = []
    for user, answer in reversed(list(zip(history[::2], history[1::2]))):
        previous = answer.get("metadata", {}).get("intent", "new_query")
        if previous in {"greeting", "out_of_scope", "clarify"}:
            break
        pairs.append((user, answer))
        if previous != "followup":
            break
    pairs.reverse()
    return pairs


def formulate_retrieval_query(message, history, intent):
    if intent != "followup":
        return retrieval_wording(message)
    pairs = topic_pairs(history)
    if not pairs:
        return retrieval_wording(message)
    context = [user["content"] for user, _answer in pairs]
    parts = [context[0], *context[1:][-2:], "Latest update (supersedes earlier constraints): " + message]
    if _refers_to_prior_reply(message.casefold()):
        suggestion = (pairs[-1][1].get("content") or "").strip()
        if suggestion:
            parts.insert(-1, "Prior suggestion being referred to: " + suggestion[:500])
    return retrieval_wording("\n".join(parts))


def horizon_phrase(text):
    closing = _CLOSING.search(text or "")
    if closing:
        return ("closing", closing.group(0))
    opening = _OPEN.search(text or "")
    if opening:
        return ("open", opening.group(0))
    return None


def situation(message, history, intent):
    """The decision in the user's words, plus deadline facts later turns replace."""
    if intent != "followup":
        horizon = horizon_phrase(message)
        return {
            "decision": message.strip(),
            "latest": message.strip(),
            "horizon": horizon,
            "previous_horizon": None,
            "support_text": message.strip(),
            "followup": False,
        }
    pairs = topic_pairs(history)
    decision = pairs[0][0]["content"].strip() if pairs else message.strip()
    horizons = []
    for user, _answer in pairs:
        found = horizon_phrase(user["content"])
        if found:
            horizons.append(found)
    found = horizon_phrase(message)
    if found:
        horizons.append(found)
    return {
        "decision": decision,
        "latest": message.strip(),
        "horizon": horizons[-1] if horizons else None,
        "previous_horizon": horizons[-2] if len(horizons) > 1 else None,
        "support_text": decision + "\n" + message.strip(),
        "followup": True,
    }

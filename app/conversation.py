"""Rule-based conversation state.

Intent labels, search rewrites, and timed facts are rules. They do not choose
a passage. A time phrase is attached to the event named in its clause, and a
later negation replaces that fact. Search wording stays out of the reply.
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
    r"\b(this afternoon|today|tonight|this morning|right now|immediately|tomorrow|in (?:an|\d+) hours?)\b",
    re.I,
)
_OPEN = re.compile(r"\b((?:a|one|1) week|next week|this week|in a week)\b", re.I)
_DECISION_WORD = re.compile(r"\b(should|whether|decide|decision|deadline)\b")
_NEGATION = re.compile(r"\b(?:isn'?t|is not|aren'?t|are not|not|no longer|anymore)\b", re.I)
_BARE_CHOICE = re.compile(r"^should (?:i|we) (?P<body>[^.?]{1,80})\??$")
_REASON_OR_CONSTRAINT = re.compile(
    r"\b(because|so that|deadline|week|today|tonight|tomorrow|afternoon|disagree|budget|funds|between|team)\b",
    re.I,
)
_REMEDY = re.compile(
    r"\b(remedy|rollback|window closes|expires|expire|becomes impossible|lose the option)\b",
    re.I,
)


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
    if _pending_fact(history) and not _different_choice(words, history):
        return "followup"
    if history and (_FOLLOWUP.match(words) or _CORRECTION.match(words) or _refers_to_prior_reply(words)):
        return "followup"
    if not history and re.search(r"\b(?:release|do|ship|launch|change|decide) it\b", words):
        return "clarify"
    if bare_choice(text):
        return "clarify"
    return "new_query"


def _pending_fact(history):
    if not history or history[-1].get("role") != "assistant":
        return False
    return (history[-1].get("metadata") or {}).get("clarify_kind") == "missing_fact"


def _different_choice(words, history):
    """A new 'should I …' question leaves the clarification that was in progress."""
    match = _BARE_CHOICE.match(words)
    if not match:
        return False
    pending = ((history[-1].get("metadata") or {}).get("plan") or {}).get("decision", "").casefold()
    body = match.group("body")
    tokens = [token for token in re.findall(r"[a-z]{4,}", body) if token not in {"should", "change"}]
    return not any(token in pending for token in tokens)


def _refers_to_prior_reply(words):
    return bool(re.search(
        r"\b(you said|you suggested|your suggestion|the first one|the second one|do that|go with that)\b",
        words,
    ))


def bare_choice(text):
    """A yes/no choice with no reason and no constraint. Template input, not a passage."""
    match = _BARE_CHOICE.match(text.strip().casefold().rstrip("."))
    if not match:
        return None
    body = match.group("body").strip()
    if _REASON_OR_CONSTRAINT.search(body) or len(body.split()) > 8:
        return None
    return body


def reason_question(body):
    """Template question for a bare choice. Employment exits get a readable form."""
    if re.search(r"\b(change jobs|leave (?:my |this |the )?job|quit(?: my job)?)\b", body):
        return "What's the main reason you're considering leaving your current job?"
    return "What's the main reason you're considering " + body.rstrip("?") + "?"


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
    choice = bare_choice(message)
    if choice and intent == "clarify":
        return reason_question(choice)
    if intent == "greeting" or _HELP_ONLY.fullmatch(words.strip(" .!?")) or _HELP_DECIDE.fullmatch(words.strip(" .!?")):
        return WELCOME_QUESTION
    if re.search(r"\b(?:release|do|ship|launch) it\b", words):
        return "What is that decision about?"
    if needs_clarification(message, intent):
        return "Which option do you mean?"
    return WELCOME_QUESTION


def remedy_question(when):
    """Template. A deadline and an expiring remedy are different facts."""
    return (
        f"Does {when} mean a remedy disappears if you wait, "
        "or is that only when the decision is due?"
    )


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
        meta = answer.get("metadata") or {}
        previous = meta.get("intent", "new_query")
        kind = meta.get("clarify_kind") or ""
        if previous in {"greeting", "out_of_scope"} or (previous == "clarify" and kind != "missing_fact"):
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


def _clauses(text):
    return [clause.strip() for clause in re.split(r"\b(?:but|however)\b|[.;]", text or "", flags=re.I) if clause.strip()]


def _event_kind(clause):
    folded = clause.casefold()
    if re.search(r"\bmeet(?:ing)?\b", folded) and not re.search(r"\b(decision|release|deadline)\b", folded):
        return "meeting"
    if _REMEDY.search(folded):
        return "remedy"
    if re.search(r"\b(decision|release|deadline|feature|due|decide)\b", folded):
        return "decision"
    return "unspecified"


def timed_events(text):
    """Each time phrase stays with the event in its own clause."""
    events = []
    for clause in _clauses(text):
        when = horizon_phrase(clause)
        if not when:
            continue
        events.append({
            "event": _event_kind(clause),
            "when": when,
            "negated": bool(_NEGATION.search(clause)),
        })
    return events


def _same_when(left, right):
    if not left or not right:
        return False
    return left[1].casefold() == right[1].casefold()


def apply_timed_facts(messages):
    """Walk the topic in order. A negation clears the deadline it names."""
    decision_deadline = None
    previous_deadline = None
    meeting = None
    remedy = False
    for message in messages:
        if _REMEDY.search(message):
            remedy = True
        for event in timed_events(message):
            when = event["when"]
            if event["negated"]:
                if decision_deadline and (_same_when(decision_deadline, when) or event["event"] in {"decision", "unspecified"}):
                    previous_deadline = decision_deadline
                    decision_deadline = None
                continue
            if event["event"] == "meeting":
                meeting = when
                continue
            if event["event"] == "remedy":
                remedy = True
            if decision_deadline and not _same_when(decision_deadline, when):
                previous_deadline = decision_deadline
            decision_deadline = when
    return {
        "decision_deadline": decision_deadline,
        "previous_deadline": previous_deadline,
        "meeting": meeting,
        "remedy_claimed": remedy,
    }


def situation(message, history, intent):
    """Facts for this topic only. A new topic does not inherit the previous one."""
    if intent != "followup":
        timed = apply_timed_facts([message])
        return {
            "decision": message.strip(),
            "latest": message.strip(),
            "support_text": message.strip(),
            "user_text": message.strip(),
            "followup": False,
            "supplied": "",
            **timed,
        }
    pairs = topic_pairs(history)
    decision = pairs[0][0]["content"].strip() if pairs else message.strip()
    messages = [user["content"] for user, _answer in pairs] + [message.strip()]
    timed = apply_timed_facts(messages)
    supplied = ""
    if not timed_events(message) and not _REMEDY.search(message):
        supplied = re.sub(r"^(?:no[.!]?\s+)", "", message.strip(), flags=re.I)
    return {
        "decision": decision,
        "latest": message.strip(),
        "support_text": decision + "\n" + message.strip(),
        "user_text": "\n".join(messages),
        "followup": True,
        "supplied": supplied,
        **timed,
    }

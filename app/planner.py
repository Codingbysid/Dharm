"""In-house decision planner.

RC1 still retrieves, abstains, and verifies ordered pairs. This module does not
change those scores.

Selection rules, all explicit:
- The ranker's primary is the action unless the user has given evidence for the
  competing passage's own flip condition.
- Words that appear only in an opposing condition are not positive support.
  Support words come from the principle, cues, domain, translation, and context.
- A deadline alone does not prove that a remedy expires. If that missing fact
  would switch to the competing passage, ask one question instead of flipping.
- A second step is included only when the pair verifier accepted that order and
  the action is still the first passage of that pair.
- A bare yes/no choice asks for the missing reason. Abstention is reserved for
  an out-of-scope request or an RC1 abstention.

The reply text is a template assembled from those fields.
"""
from dataclasses import asdict, dataclass, field
import re

from app.conversation import (
    bare_choice,
    clarifying_question,
    needs_clarification,
    remedy_question,
    situation,
)

_WORD = re.compile(r"[a-z]{4,}")
_STOP = {
    "about", "after", "again", "also", "because", "before", "change", "could", "decide",
    "decision", "doing", "first", "from", "have", "help", "just", "keep", "make",
    "more", "need", "should", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "want", "what", "when", "where", "which", "with",
    "would", "your", "into", "over", "under", "than", "still", "please", "whether",
    "back", "company", "last", "quarter",
}
_TIME = {
    "today", "tonight", "tomorrow", "afternoon", "morning", "week", "deadline",
    "immediately", "hour", "hours", "right",
}
_EXPIRY = re.compile(r"remedy|cutoff|expir|waiting|impossible", re.I)
_MISCONDUCT = re.compile(r"misconduct|wrongdoing|conceal|crime|fraud|abuse|illegal", re.I)
_EDITORIAL = re.compile(r"sattvic|editorial label", re.I)


@dataclass
class Support:
    id: str
    principle: str
    role: str


@dataclass
class DecisionPlan:
    disposition: str
    decision: str
    facts: list = field(default_factory=list)
    action: str = ""
    reason: str = ""
    supports: list = field(default_factory=list)
    limit: str = ""
    second_action: str = ""
    clarify_kind: str = ""

    def as_dict(self):
        payload = asdict(self)
        payload["supports"] = [asdict(item) if isinstance(item, Support) else item for item in self.supports]
        return payload


def content_words(text):
    return {word for word in _WORD.findall((text or "").casefold()) if word not in _STOP}


def support_words(record):
    """Positive support. The competing condition is excluded on purpose."""
    parts = [
        record.get("derived_principle"),
        record.get("retrieval_cues"),
        record.get("decision_domain"),
        record.get("english_translation"),
        record.get("context_summary"),
    ]
    return content_words(" ".join(part for part in parts if part))


def condition_evidence(user_text, condition):
    """Specific overlap with a flip condition, ignoring a bare time word."""
    if not condition:
        return set()
    return (content_words(user_text) - _TIME) & (content_words(condition) - _TIME)


def lead_sentence(text):
    text = (text or "").strip()
    if not text:
        return ""
    return re.split(r"(?<=[.])\s+", text, maxsplit=1)[0].strip()


def _facts(state):
    facts = []
    deadline = state.get("decision_deadline")
    previous = state.get("previous_deadline")
    if previous and deadline and not _same(previous, deadline):
        facts.append(f"decision deadline replaced: {previous[1]} -> {deadline[1]}")
    elif deadline:
        facts.append(f"decision deadline: {deadline[1]}")
    elif previous:
        facts.append(f"decision deadline cleared: {previous[1]}")
    meeting = state.get("meeting")
    if meeting:
        facts.append(f"meeting: {meeting[1]}")
    if state.get("remedy_claimed"):
        facts.append("remedy expires")
    if state.get("supplied"):
        facts.append("update: " + state["supplied"])
    return facts


def _same(left, right):
    return left and right and left[1].casefold() == right[1].casefold()


def _plain_reason(state):
    """Short explanation from facts. Does not repeat the user's whole question."""
    deadline = state.get("decision_deadline")
    previous = state.get("previous_deadline")
    meeting = state.get("meeting")
    if state.get("supplied"):
        bits = [state["supplied"].rstrip(".")]
    elif meeting and deadline:
        bits = [f"The meeting is {meeting[1]}", f"the decision is due {deadline[1]}"]
    else:
        decision = state.get("decision") or ""
        first = re.split(r"(?<=[.?!])\s+", decision.strip(), maxsplit=1)[0]
        first = re.sub(
            r"^(?:help me (?:choose|decide) whether to|should i|should we)\s+",
            "", first, flags=re.I,
        )
        first = re.sub(r"\bwhat should i do first\??", "", first, flags=re.I).strip(" .?")
        if first.lower().startswith("my "):
            first = "Your " + first[3:]
        bits = []
        if first and " but " not in first.casefold() and not re.match(r"what should i\b", first, re.I):
            bits.append(first[0].upper() + first[1:])
        if previous and deadline and not _same(previous, deadline):
            bits.append(f"the decision is now due {deadline[1]}, replacing {previous[1]}")
        elif deadline and (not first or deadline[1].casefold() not in first.casefold()):
            bits.append(f"the decision is due {deadline[1]}")
    if previous and deadline and not _same(previous, deadline) and state.get("supplied"):
        bits.append(f"the decision is now due {deadline[1]}, replacing {previous[1]}")
    if state.get("remedy_claimed") and not any("remedy" in bit.casefold() for bit in bits):
        bits.append("waiting would close a remedy")
    if not bits:
        return "This is the first step that fits what you described."
    sentence = bits[0]
    if len(bits) > 1:
        sentence = sentence.rstrip(".") + ", and " + ", and ".join(bits[1:])
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _limit(record, user_text):
    """The condition that would leave the selected passage, not the reason for staying."""
    edge = record.get("competing_passage_id") or {}
    text = (edge.get("flip_to_competing_when") or "").strip().rstrip(".")
    if not text or _EDITORIAL.search(text):
        return ""
    if _MISCONDUCT.search(text) and not _MISCONDUCT.search(user_text or ""):
        return ""
    return text


def _clarify(state, question, kind):
    return DecisionPlan(
        disposition="clarify",
        decision=state["decision"],
        facts=_facts(state),
        action=question,
        reason="",
        clarify_kind=kind,
    )


def _abstain(state):
    decision = (state.get("decision") or "this").strip().rstrip(".?")
    return DecisionPlan(
        disposition="abstain",
        decision=state["decision"],
        facts=_facts(state),
        reason=f"These texts do not settle this: {decision}.",
    )


def _assumes_expiry(record):
    edge = record.get("competing_passage_id") or {}
    blob = " ".join([record.get("derived_principle") or "", edge.get("retain_self_when") or ""])
    return bool(_EXPIRY.search(blob))


def _expiry_unresolved(record, top3, state):
    """A closing deadline is not evidence that a remedy expires."""
    deadline = state.get("decision_deadline")
    if not deadline or deadline[0] != "closing" or state.get("remedy_claimed"):
        return None
    edge = record.get("competing_passage_id") or {}
    neighbor = edge.get("id")
    flip = edge.get("flip_to_competing_when") or ""
    neighbor_would_change = neighbor in top3 and _EXPIRY.search(flip) and not condition_evidence(state.get("user_text"), flip)
    if neighbor_would_change or _assumes_expiry(record):
        return deadline[1]
    return None


def build_plan(message, history, intent, retrieval, records):
    state = situation(message, history, intent)
    if bare_choice(message) and intent != "followup":
        return _clarify(state, clarifying_question(message, "clarify"), "missing_fact")
    if intent in {"greeting", "clarify"} or needs_clarification(message, intent):
        kind = "missing_fact" if bare_choice(message) else "open"
        if needs_clarification(message, intent):
            kind = "open"
        return _clarify(state, clarifying_question(message, intent if intent != "new_query" else "clarify"), kind)
    if intent == "out_of_scope" or retrieval.get("abstained") or retrieval.get("primary") in {None, "Abstain"}:
        return _abstain(state)

    top3 = [pid for pid in (retrieval.get("top3") or []) if pid in records]
    primary = retrieval.get("primary")
    if primary not in records:
        return _abstain(state)

    unresolved = _expiry_unresolved(records[primary], top3, state)
    if unresolved:
        return _clarify(state, remedy_question(unresolved), "missing_fact")

    chosen = primary
    flipped = False
    edge = records[primary].get("competing_passage_id") or {}
    neighbor = edge.get("id")
    if neighbor in top3 and neighbor in records and condition_evidence(state.get("user_text"), edge.get("flip_to_competing_when")):
        chosen = neighbor
        flipped = True

    record = records[chosen]
    action = lead_sentence(record.get("derived_principle") or "")
    supports = [Support(chosen, action, "primary")]
    promoted = [pid for pid in (retrieval.get("promoted_pair") or []) if pid in records] if retrieval.get("pair_promoted") else []
    second = ""
    if len(promoted) == 2 and not flipped and chosen == promoted[0]:
        second = lead_sentence(records[promoted[1]].get("derived_principle") or "")
        supports.append(Support(promoted[1], second, "second"))
    return DecisionPlan(
        disposition="answer",
        decision=state["decision"],
        facts=_facts(state),
        action=action,
        reason=_plain_reason(state),
        supports=supports,
        limit=_limit(record, state.get("user_text")),
        second_action=second,
    )

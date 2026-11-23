"""In-house decision planner.

RC1 still retrieves, abstains, and verifies ordered pairs. This module does not
change those scores. It decides whether a retrieved passage has earned an
answer.

The choice rules are explicit:
- A greeting or an unnamed decision asks one question.
- An abstention from RC1, an out-of-scope request, or a top-three set that
  neither shares the user's words nor was accepted as an ordered pair becomes
  an abstention.
- Otherwise the passage in the top three that shares the most of the user's
  own words is the action. A tie keeps RC1's order.
- A closing deadline can move the action to that passage's competing neighbor
  when the neighbor was retrieved and its flip condition is about a cutoff.
- A second action is included only when the pair verifier accepted that order
  and the deadline rule did not replace it.

Corpus scenarios are not treated as facts about the user.
"""
from dataclasses import asdict, dataclass, field
import re

from app.conversation import clarifying_question, needs_clarification, situation

_WORD = re.compile(r"[a-z]{4,}")
_STOP = {
    "about", "after", "again", "also", "because", "before", "change", "could", "decide",
    "decision", "doing", "first", "from", "have", "help", "just", "keep", "make",
    "more", "need", "should", "that", "their", "them", "then", "there", "these",
    "they", "this", "those", "want", "what", "when", "where", "which", "with",
    "would", "your", "into", "over", "under", "than", "still", "please", "whether",
    "back", "company", "last", "quarter",
}
_CUTOFF = ("cutoff", "urgent", "expir", "delay", "immediate")


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

    def as_dict(self):
        payload = asdict(self)
        payload["supports"] = [asdict(item) if isinstance(item, Support) else item for item in self.supports]
        return payload


def content_words(text):
    return {word for word in _WORD.findall((text or "").casefold()) if word not in _STOP}


def passage_words(record):
    edge = record.get("competing_passage_id") or {}
    parts = [
        record.get("derived_principle"),
        record.get("retrieval_cues"),
        record.get("decision_domain"),
        record.get("english_translation"),
        record.get("context_summary"),
        edge.get("condition"),
    ]
    return content_words(" ".join(part for part in parts if part))


def _facts(state):
    facts = []
    horizon = state.get("horizon")
    previous = state.get("previous_horizon")
    if previous and horizon and previous[1].casefold() != horizon[1].casefold():
        facts.append(f"deadline replaced: {previous[1]} -> {horizon[1]}")
    elif horizon:
        facts.append(f"deadline: {horizon[1]}")
    if state.get("followup"):
        facts.append("latest: " + state["latest"])
    return facts


def _spoken_decision(state):
    decision = state["decision"].strip().rstrip(".")
    if decision.endswith("?"):
        decision = decision[:-1].rstrip()
    return decision


def _reason(state):
    decision = _spoken_decision(state)
    reason = f"The decision is: {decision}."
    horizon = state.get("horizon")
    previous = state.get("previous_horizon")
    if previous and horizon and previous[1].casefold() != horizon[1].casefold():
        reason += f" The deadline is now {horizon[1]}, replacing {previous[1]}."
    elif horizon:
        reason += f" The deadline to use is {horizon[1]}."
    if state.get("followup"):
        reason += " Use this update: " + state["latest"]
    return reason


def _limit(record, flipped):
    edge = record.get("competing_passage_id") or {}
    text = edge.get("retain_self_when") if flipped else edge.get("flip_to_competing_when")
    text = (text or record.get("application_limit") or "").strip()
    return text.rstrip(".")


def _flip_target(record, top3, state):
    horizon = state.get("horizon")
    if not horizon or horizon[0] != "closing":
        return None
    edge = record.get("competing_passage_id") or {}
    neighbor = edge.get("id")
    flip = (edge.get("flip_to_competing_when") or "").casefold()
    if neighbor in top3 and any(token in flip for token in _CUTOFF):
        return neighbor
    return None


def _abstain(state, reason):
    decision = _spoken_decision(state)
    if reason == "These texts do not settle that.":
        spoken = f"These texts do not settle this: {decision}."
    else:
        spoken = f"These texts do not settle this: {decision}. Nothing retrieved is close enough to support a recommendation."
    return DecisionPlan(
        disposition="abstain",
        decision=state["decision"],
        facts=_facts(state),
        action="",
        reason=spoken,
        limit="",
    )


def build_plan(message, history, intent, retrieval, records):
    state = situation(message, history, intent)
    if intent in {"greeting", "clarify"} or needs_clarification(message, intent):
        return DecisionPlan(
            disposition="clarify",
            decision=state["decision"],
            facts=[],
            action=clarifying_question(message, intent),
            reason="",
        )
    if intent == "out_of_scope" or retrieval.get("abstained") or retrieval.get("primary") in {None, "Abstain"}:
        return _abstain(state, "These texts do not settle that.")
    top3 = [pid for pid in (retrieval.get("top3") or []) if pid in records]
    if not top3:
        return _abstain(state, "These texts do not settle that.")

    user_words = content_words(state["support_text"])
    overlaps = {pid: user_words & passage_words(records[pid]) for pid in top3}
    best_size = max(len(words) for words in overlaps.values())
    promoted = list(retrieval.get("promoted_pair") or []) if retrieval.get("pair_promoted") else []
    promoted = [pid for pid in promoted if pid in records]
    flipped = False
    if best_size > 0:
        chosen = next(pid for pid in top3 if len(overlaps[pid]) == best_size)
        neighbor = _flip_target(records[chosen], top3, state)
        if neighbor and neighbor in records:
            chosen = neighbor
            flipped = True
    elif len(promoted) == 2:
        chosen = promoted[0]
    else:
        return _abstain(
            state,
            "These texts do not settle that. Nothing retrieved is close enough to this decision to support a recommendation.",
        )

    record = records[chosen]
    supports = [Support(chosen, record.get("derived_principle") or "", "primary")]
    second = ""
    if len(promoted) == 2 and not flipped and chosen == promoted[0]:
        other = records[promoted[1]]
        second = other.get("derived_principle") or ""
        supports.append(Support(promoted[1], second, "second"))
    return DecisionPlan(
        disposition="answer",
        decision=state["decision"],
        facts=_facts(state),
        action=record.get("derived_principle") or "",
        reason=_reason(state),
        supports=supports,
        limit=_limit(record, flipped),
        second_action=second,
    )

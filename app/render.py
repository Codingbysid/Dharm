"""Template renderer for a DecisionPlan.

Every sentence the user sees with Gemini switched off is assembled here from
fields the planner already approved. It does not add a passage the planner
rejected, and it does not print passage IDs. A replaced deadline is explained
once, as why the latest fact still supports the selected action.
"""
import re

from app.conversation import TRADITION, WELCOME_QUESTION

_REPLACED = re.compile(r"^decision deadline replaced:\s*(.+?)\s*->\s*(.+)$")


def _clause(text):
    text = text.strip().rstrip(".")
    if not text:
        return ""
    return text[0].lower() + text[1:]


def _replaced_deadline_reason(plan):
    """One sentence. The planner's action and citations stay as decided."""
    old = new = None
    update = ""
    for fact in plan.facts or []:
        match = _REPLACED.match(fact)
        if match:
            old, new = match.group(1).strip(), match.group(2).strip()
        elif fact.startswith("update:"):
            update = fact.split(":", 1)[1].strip()
    if not old or not new:
        return None
    clauses = [part.strip(" .") for part in re.split(r"[.?!]", update) if part.strip(" .")]
    if not clauses:
        return f"The decision is now due {new} instead of {old}, so this step still fits."
    first = _clause(clauses[0])
    if any(re.search(r"only when the decision is due", clause, re.I) for clause in clauses[1:]):
        return (
            f"Because {first}, and {new} is only when the decision is due, "
            f"instead of {old}, this step still fits."
        )
    if len(clauses) > 1:
        return (
            f"Because {first} and {_clause(clauses[1])}, the decision is now due {new} "
            f"instead of {old}, so this step still fits."
        )
    return f"Because {first}, the decision is now due {new} instead of {old}, so this step still fits."


def render_plan(plan):
    if plan.disposition == "clarify":
        question = plan.action or WELCOME_QUESTION
        if question == WELCOME_QUESTION:
            return question + "\n\n" + TRADITION
        return question
    if plan.disposition == "abstain":
        return plan.reason or "These texts do not settle that."
    parts = [plan.action]
    if plan.second_action:
        second = plan.second_action[0].lower() + plan.second_action[1:]
        parts.append("Then " + second)
    reason = _replaced_deadline_reason(plan) or plan.reason
    if reason:
        parts.append(reason)
    if plan.limit:
        limit = plan.limit[0].lower() + plan.limit[1:]
        parts.append("This would change if " + limit + ".")
    return "\n\n".join(part for part in parts if part).strip()

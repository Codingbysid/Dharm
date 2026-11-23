"""Template renderer for a DecisionPlan.

Every sentence the user sees with Gemini switched off is assembled here from
fields the planner already approved. It does not add a passage the planner
rejected, and it does not print passage IDs.
"""
from app.conversation import TRADITION, WELCOME_QUESTION


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
        parts.append("Then: " + plan.second_action)
    if plan.reason:
        parts.append(plan.reason)
    if plan.limit:
        parts.append("This would change if: " + plan.limit)
    return "\n\n".join(part for part in parts if part).strip()

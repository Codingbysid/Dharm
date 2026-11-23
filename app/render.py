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
        second = plan.second_action[0].lower() + plan.second_action[1:]
        parts.append("Then " + second)
    if plan.reason:
        parts.append(plan.reason)
    if plan.limit:
        limit = plan.limit[0].lower() + plan.limit[1:]
        parts.append("This would change if " + limit + ".")
    return "\n\n".join(part for part in parts if part).strip()

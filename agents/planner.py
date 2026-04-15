from __future__ import annotations

import json
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from config import PLANNER_MODEL
from models.schemas import AgentState, Plan


class PlannerAgent(BaseAgent):
    model_name: ClassVar[str] = PLANNER_MODEL
    max_tokens_key: ClassVar[str] = "planner"
    prompt_name: ClassVar[str] = "planner"


_agent: PlannerAgent | None = None


def _get_agent() -> PlannerAgent:
    global _agent
    if _agent is None:
        _agent = PlannerAgent()
    return _agent


def planner_node(state: AgentState) -> dict[str, Any]:
    """Produces a structured implementation plan for the user's request."""
    agent = _get_agent()
    llm = agent.llm.with_structured_output(Plan)

    messages = [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=f"Create an implementation plan for: {state.user_request}"),
    ]

    plan: Plan = llm.invoke(messages)  # type: ignore[assignment]

    return {
        "plan": plan,
        "messages": [
            HumanMessage(content=f"Plan created:\n{json.dumps(plan.model_dump(), indent=2)}")
        ],
    }

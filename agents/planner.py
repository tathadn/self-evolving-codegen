from __future__ import annotations

import json
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import MAX_TOKENS, PLANNER_MODEL
from models.schemas import AgentState, Plan

_PROMPT = (Path(__file__).parent.parent / "prompts" / "planner.md").read_text()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=PLANNER_MODEL,
        max_tokens=MAX_TOKENS["planner"],
    )


def planner_node(state: AgentState) -> dict:
    """Produces a structured implementation plan for the user's request."""
    llm = get_llm().with_structured_output(Plan)

    messages = [
        SystemMessage(content=_PROMPT),
        HumanMessage(content=f"Create an implementation plan for: {state.user_request}"),
    ]

    plan: Plan = llm.invoke(messages)  # type: ignore[assignment]

    return {
        "plan": plan,
        "messages": [
            HumanMessage(content=f"Plan created:\n{json.dumps(plan.model_dump(), indent=2)}")
        ],
    }

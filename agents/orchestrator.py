from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from config import ORCHESTRATOR_MODEL
from models.schemas import AgentState, TaskStatus


class OrchestratorAgent(BaseAgent):
    model_name: ClassVar[str] = ORCHESTRATOR_MODEL
    max_tokens_key: ClassVar[str] = "orchestrator"
    prompt_name: ClassVar[str] = "orchestrator"


_agent: OrchestratorAgent | None = None


def _get_agent() -> OrchestratorAgent:
    global _agent
    if _agent is None:
        _agent = OrchestratorAgent()
    return _agent


def orchestrator_node(state: AgentState) -> dict[str, Any]:
    """Entry point: interprets the user request and sets the initial status."""
    agent = _get_agent()

    messages = [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(
            content=(
                f"User request: {state.user_request}\n\n"
                f"Current iteration: {state.iteration}/{state.max_iterations}\n"
                f"Status: {state.status}"
            )
        ),
    ]

    response = agent.llm.invoke(messages)

    return {
        "messages": [response],
        "status": TaskStatus.IN_PROGRESS,
    }


def should_continue(state: AgentState) -> str:
    """Route after review + test: continue, revise, or finish."""
    if state.status == TaskStatus.COMPLETED:
        return "end"

    if state.status == TaskStatus.FAILED:
        return "end"

    if state.iteration >= state.max_iterations:
        return "end"

    review = state.review
    test = state.test_result

    if review and review.approved and test and test.passed:
        return "end"

    if state.iteration < state.max_iterations:
        return "revise"

    return "end"

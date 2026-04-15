from __future__ import annotations

from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent
from config import REVIEWER_MODEL
from models.schemas import AgentState, ReviewFeedback, TaskStatus


class ReviewerAgent(BaseAgent):
    model_name: ClassVar[str] = REVIEWER_MODEL
    max_tokens_key: ClassVar[str] = "reviewer"
    prompt_name: ClassVar[str] = "reviewer"


_agent: ReviewerAgent | None = None


def _get_agent() -> ReviewerAgent:
    global _agent
    if _agent is None:
        _agent = ReviewerAgent()
    return _agent


def _format_artifacts(state: AgentState) -> str:
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)


def reviewer_node(state: AgentState) -> dict[str, Any]:
    """Reviews generated code and returns structured feedback."""
    agent = _get_agent()
    llm = agent.llm.with_structured_output(ReviewFeedback)

    messages = [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=_format_artifacts(state)),
    ]

    review: ReviewFeedback = llm.invoke(messages)  # type: ignore[assignment]

    status = TaskStatus.COMPLETED if review.approved else TaskStatus.NEEDS_REVISION
    summary = HumanMessage(
        content=(
            f"Review complete — score: {review.score}/10, "
            f"approved: {review.approved}. {review.summary}"
        )
    )

    return {
        "review": review,
        "status": status,
        "messages": [summary],
    }

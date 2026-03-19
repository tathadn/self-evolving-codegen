from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import MAX_TOKENS, REVIEWER_MODEL
from models.schemas import AgentState, ReviewFeedback, TaskStatus

_PROMPT = (Path(__file__).parent.parent / "prompts" / "reviewer.md").read_text()


def get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=REVIEWER_MODEL,
        max_tokens=MAX_TOKENS["reviewer"],
    )


def _format_artifacts(state: AgentState) -> str:
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)


def reviewer_node(state: AgentState) -> dict:
    """Reviews generated code and returns structured feedback."""
    llm = get_llm().with_structured_output(ReviewFeedback)

    messages = [
        SystemMessage(content=_PROMPT),
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

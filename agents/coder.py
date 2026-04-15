from __future__ import annotations

import json
from typing import Any, ClassVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.base import BaseAgent
from config import CODER_MODEL
from models.schemas import AgentState, CodeArtifact


class ArtifactList(BaseModel):
    artifacts: list[CodeArtifact]


class CoderAgent(BaseAgent):
    model_name: ClassVar[str] = CODER_MODEL
    max_tokens_key: ClassVar[str] = "coder"
    prompt_name: ClassVar[str] = "coder"


_agent: CoderAgent | None = None


def _get_agent() -> CoderAgent:
    global _agent
    if _agent is None:
        _agent = CoderAgent()
    return _agent


def _build_prompt(state: AgentState) -> str:
    parts = [f"User request: {state.user_request}"]

    if state.plan:
        parts.append(f"\nImplementation plan:\n{json.dumps(state.plan.model_dump(), indent=2)}")

    if state.review and not state.review.approved:
        issues = "\n".join(f"- {i}" for i in state.review.issues)
        parts.append(f"\nReview issues to fix:\n{issues}")

    if state.test_result and not state.test_result.passed:
        parts.append("\nTest execution errors to fix (real sandbox output):")
        if state.test_result.errors:
            for err in state.test_result.errors:
                parts.append(err)
        if state.test_result.output:
            parts.append(f"\nFull pytest output:\n{state.test_result.output}")

    if state.artifacts:
        parts.append("\nExisting code to revise:")
        for artifact in state.artifacts:
            parts.append(
                f"\n### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```"
            )

    return "\n".join(parts)


def coder_node(state: AgentState) -> dict[str, Any]:
    """Generates or revises code artifacts based on the plan and feedback."""
    agent = _get_agent()
    llm = agent.llm.with_structured_output(ArtifactList)

    messages = [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=_build_prompt(state)),
    ]

    result: ArtifactList = llm.invoke(messages)  # type: ignore[assignment]

    filenames = [a.filename for a in result.artifacts]
    summary = HumanMessage(
        content=f"Code generated: {len(result.artifacts)} file(s) — {', '.join(filenames)}"
    )

    return {
        "artifacts": result.artifacts,
        "messages": [summary],
        "iteration": state.iteration + 1,
    }

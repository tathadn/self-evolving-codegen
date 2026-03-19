from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

from config import MAX_TOKENS, TESTER_MODEL
from models.schemas import AgentState, CodeArtifact, TaskStatus, TestResult
from sandbox.runner import CodeFile, run_in_sandbox

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(generation: int) -> str:
    """Load the tester system prompt for the given generation.

    Generation 0 uses the original ``prompts/tester.md``.
    Generation N > 0 uses ``prompts/tester_gen_{N}.txt``.
    """
    if generation == 0:
        return (_PROMPTS_DIR / "tester.md").read_text()
    path = _PROMPTS_DIR / f"tester_gen_{generation}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Evolved prompt for generation {generation} not found at {path}. "
            "Run the evolution loop first or use generation=0."
        )
    return path.read_text()


class TestFileList(BaseModel):
    artifacts: list[CodeArtifact] = Field(default_factory=list)

    @field_validator("artifacts", mode="before")
    @classmethod
    def parse_if_string(cls, v: object) -> object:
        if isinstance(v, str):
            return json.loads(v)
        return v


def get_llm() -> ChatAnthropic:
    """Return the LLM used by the tester."""
    return ChatAnthropic(
        model=TESTER_MODEL,
        max_tokens=MAX_TOKENS["tester"],
    )


def _format_artifacts(state: AgentState) -> str:
    """Format code artifacts and the original request for the LLM prompt."""
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)



def make_tester_node(generation: int = 0) -> Callable[[AgentState], dict]:
    """Return a LangGraph-compatible tester node for the given prompt generation.

    Generation 0 loads the original V1 prompt and behaves identically to V1.
    Generation N > 0 loads ``prompts/tester_gen_{N}.txt`` produced by the evolver.

    Args:
        generation: Which evolved prompt version to use (default 0 = V1 behaviour).

    Returns:
        A callable ``(state: AgentState) -> dict`` suitable for use as a graph node.
    """
    prompt = _load_prompt(generation)

    def tester_node(state: AgentState) -> dict:
        """Generate test files via LLM, execute them in the sandbox, return TestResult."""
        llm = get_llm().with_structured_output(TestFileList)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=_format_artifacts(state)),
        ]

        test_file_list: TestFileList = llm.invoke(messages)  # type: ignore[assignment]

        if not test_file_list.artifacts:
            result = TestResult(
                passed=False,
                errors=["Tester agent did not generate any test files."],
                output="",
                generation=generation,
            )
            return {
                "test_result": result,
                "status": TaskStatus.NEEDS_REVISION,
                "messages": [HumanMessage(content="No test files were generated.")],
            }

        # Combine source + test files and run in sandbox
        code_files = [CodeFile(filename=a.filename, content=a.content) for a in state.artifacts]
        test_files = [
            CodeFile(filename=a.filename, content=a.content) for a in test_file_list.artifacts
        ]
        raw_test_code = "\n\n".join(a.content for a in test_file_list.artifacts)
        requirements = state.plan.dependencies if state.plan else None

        base_result = run_in_sandbox(code_files + test_files, requirements=requirements or None)

        # Attach V2 evolution metadata the runner doesn't know about
        result = base_result.model_copy(
            update={"generation": generation, "test_code": raw_test_code}
        )

        status = TaskStatus.COMPLETED if result.passed else TaskStatus.NEEDS_REVISION
        summary = HumanMessage(
            content=(
                f"Tests: {result.passed_tests}/{result.total_tests} passed. "
                f"{'All tests passed.' if result.passed else f'Failures: {result.failed_tests}'}"
            )
        )

        return {
            "test_result": result,
            "status": status,
            "messages": [summary],
        }

    return tester_node


# Backward-compatible alias: generation=0 behaves identically to V1
tester_node = make_tester_node(generation=0)

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator

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
        model=os.getenv("TESTER_MODEL", "claude-sonnet-4-6"),
        max_tokens=4096,
    )


def _format_artifacts(state: AgentState) -> str:
    """Format code artifacts and the original request for the LLM prompt."""
    parts = [f"Original request: {state.user_request}\n"]
    for artifact in state.artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```\n")
    return "\n".join(parts)


def _parse_pytest_counts(stdout: str) -> tuple[int, int, int]:
    """Extract total/passed/failed counts from pytest summary line."""
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    passed_count = int(passed_match.group(1)) if passed_match else 0
    failed_count = int(failed_match.group(1)) if failed_match else 0
    total = passed_count + failed_count
    return total, passed_count, failed_count


def _parse_per_test_results(stdout: str) -> list[dict]:
    """Parse per-test PASSED/FAILED lines from pytest verbose output."""
    results = []
    for line in stdout.splitlines():
        if " PASSED" in line:
            results.append({"name": line.split(" PASSED")[0].strip(), "passed": True})
        elif " FAILED" in line:
            results.append({"name": line.split(" FAILED")[0].strip(), "passed": False})
    return results


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

        sandbox_result = run_in_sandbox(code_files + test_files)

        per_test = _parse_per_test_results(sandbox_result.stdout)

        if sandbox_result.success:
            total, passed_count, failed_count = _parse_pytest_counts(sandbox_result.stdout)
            result = TestResult(
                passed=True,
                total_tests=total or len(test_files),
                passed_tests=passed_count or len(test_files),
                failed_tests=0,
                errors=[],
                output=sandbox_result.stdout,
                generation=generation,
                test_code=raw_test_code,
                per_test_results=per_test or None,
            )
        else:
            total, passed_count, failed_count = _parse_pytest_counts(sandbox_result.stdout)
            errors = [sandbox_result.stderr] if sandbox_result.stderr.strip() else []
            if sandbox_result.stdout.strip():
                errors.append(sandbox_result.stdout)
            result = TestResult(
                passed=False,
                total_tests=total or len(test_files),
                passed_tests=passed_count,
                failed_tests=failed_count or (total - passed_count),
                errors=errors,
                output=sandbox_result.stdout,
                generation=generation,
                test_code=raw_test_code,
                per_test_results=per_test or None,
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

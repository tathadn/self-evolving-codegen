from __future__ import annotations

import json
from datetime import datetime

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import EVALUATOR_MODEL, MAX_TOKENS, WEIGHTS
from evolution.models import GenerationMetrics, TestEffectivenessScore
from models.schemas import CodeArtifact, TestResult

_JUDGE_PROMPT = """You are an expert software testing evaluator acting as an impartial judge.

You will receive:
1. The original coding task (user request)
2. The generated source code artifacts
3. The generated test code
4. The test execution output

Your job is to evaluate each individual test case and score it on four dimensions.

For EACH test function you identify in the test code, produce one entry in "scores" with:
- test_name: the function name (e.g. "test_divide_by_zero")
- caught_real_bug: true if this test would catch a genuine defect in a naive implementation
- was_redundant: true if this test duplicates the coverage of another test in the file
- was_false_failure: true if this test fails for a reason unrelated to actual code correctness \
(e.g. wrong import path, environment dependency, incorrect assertion on valid output)
- coverage_category: one of "happy_path", "edge_case", "error_handling", "integration"

Then provide two aggregate scores (1–10):
- coverage_quality: breadth and depth of test coverage overall
- edge_case_coverage: how well the tests explore boundary and error conditions

Respond ONLY with JSON, no preamble, no markdown fences:
{
  "scores": [
    {
      "test_name": "test_example",
      "caught_real_bug": true,
      "was_redundant": false,
      "was_false_failure": false,
      "coverage_category": "edge_case"
    }
  ],
  "coverage_quality": 7.5,
  "edge_case_coverage": 6.0,
  "strengths": ["<one-line observation>", "<one-line observation>"],
  "weaknesses": ["<one-line observation>", "<one-line observation>"]
}"""


def get_llm() -> ChatAnthropic:
    """Return the Haiku model used as the cost-efficient LLM judge."""
    return ChatAnthropic(
        model=EVALUATOR_MODEL,
        max_tokens=MAX_TOKENS["evaluator"],
    )


def _format_task(
    user_request: str,
    artifacts: list[CodeArtifact],
    test_artifacts: list[CodeArtifact],
    test_result: TestResult,
) -> str:
    """Build the human-turn message for the judge."""
    parts = [f"## Task\n{user_request}\n"]

    parts.append("## Source Code")
    for artifact in artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```")

    parts.append("## Generated Tests")
    for artifact in test_artifacts:
        parts.append(f"### {artifact.filename}\n```{artifact.language}\n{artifact.content}\n```")

    parts.append("## Test Execution Output")
    status = "PASSED" if test_result.passed else "FAILED"
    parts.append(
        f"Status: {status} | "
        f"Total: {test_result.total_tests} | "
        f"Passed: {test_result.passed_tests} | "
        f"Failed: {test_result.failed_tests}"
    )
    if test_result.errors:
        parts.append("Errors:\n" + "\n".join(test_result.errors))
    if test_result.output:
        parts.append(f"Output:\n{test_result.output[:2000]}")  # cap to control tokens

    return "\n\n".join(parts)


def _compute_overall_score(
    bug_detection_rate: float,
    false_failure_rate: float,
    redundancy_rate: float,
    coverage_quality: float,
    edge_case_coverage: float,
) -> float:
    """Compute weighted composite score; inverted metrics are flipped before weighting."""
    # Normalise 1-10 scores to 0-1
    coverage_quality_norm = (coverage_quality - 1) / 9
    edge_case_norm = (edge_case_coverage - 1) / 9

    raw = (
        WEIGHTS["bug_detection_rate"] * bug_detection_rate
        + WEIGHTS["false_failure_rate"] * (1 - false_failure_rate)
        + WEIGHTS["coverage_quality"] * coverage_quality_norm
        + WEIGHTS["edge_case_coverage"] * edge_case_norm
        + WEIGHTS["redundancy_rate"] * (1 - redundancy_rate)
    )
    return round(min(max(raw, 0.0), 1.0), 4)


def _aggregate(
    scores: list[TestEffectivenessScore],
    coverage_quality: float,
    edge_case_coverage: float,
    strengths: list[str],
    weaknesses: list[str],
    generation: int,
) -> GenerationMetrics:
    """Compute aggregate metrics from per-test scores and judge ratings."""
    total = len(scores)
    if total == 0:
        bug_detection_rate = 0.0
        false_failure_rate = 0.0
        redundancy_rate = 0.0
    else:
        bug_detection_rate = sum(1 for s in scores if s.caught_real_bug) / total
        false_failure_rate = sum(1 for s in scores if s.was_false_failure) / total
        redundancy_rate = sum(1 for s in scores if s.was_redundant) / total

    overall_score = _compute_overall_score(
        bug_detection_rate,
        false_failure_rate,
        redundancy_rate,
        coverage_quality,
        edge_case_coverage,  # noqa: E501
    )

    return GenerationMetrics(
        generation=generation,
        bug_detection_rate=round(bug_detection_rate, 4),
        false_failure_rate=round(false_failure_rate, 4),
        redundancy_rate=round(redundancy_rate, 4),
        coverage_quality=coverage_quality,
        edge_case_coverage=edge_case_coverage,
        overall_score=overall_score,
        strengths=strengths,
        weaknesses=weaknesses,
        timestamp=datetime.utcnow(),
    )


def evaluate(
    generation: int,
    user_request: str,
    artifacts: list[CodeArtifact],
    test_artifacts: list[CodeArtifact],
    test_result: TestResult,
) -> GenerationMetrics:
    """Score the tester's output using the LLM-as-Judge rubric.

    Assesses each generated test for bug detection, false failure potential,
    redundancy, and edge case coverage. Returns aggregate GenerationMetrics
    for the given generation.

    Args:
        generation: Zero-based generation index.
        user_request: The original coding task given to the pipeline.
        artifacts: Source code artifacts produced by the Coder agent.
        test_artifacts: Test files produced by the Tester agent.
        test_result: Execution result from the sandbox runner.

    Returns:
        GenerationMetrics with per-dimension scores and qualitative observations.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=_JUDGE_PROMPT),
        HumanMessage(content=_format_task(user_request, artifacts, test_artifacts, test_result)),
    ]

    try:
        response = llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)

        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())

        scores = [TestEffectivenessScore(**s) for s in data.get("scores", [])]
        coverage_quality = float(data.get("coverage_quality", 5.0))
        edge_case_coverage = float(data.get("edge_case_coverage", 5.0))
        strengths = data.get("strengths", [])
        weaknesses = data.get("weaknesses", [])

        return _aggregate(
            scores, coverage_quality, edge_case_coverage, strengths, weaknesses, generation
        )

    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        # Fallback: return a neutral score so the evolution loop never crashes
        return GenerationMetrics(
            generation=generation,
            bug_detection_rate=0.0,
            false_failure_rate=0.0,
            redundancy_rate=0.0,
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            overall_score=0.0,
            strengths=[],
            weaknesses=[f"Evaluator failed to parse LLM response: {exc}"],
            timestamp=datetime.utcnow(),
        )

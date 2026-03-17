from __future__ import annotations

import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from evolution.models import GenerationMetrics

_ANALYZER_PROMPT = """You are an expert at analyzing AI test generation quality and identifying \
specific, actionable failure patterns.

You will receive:
1. The current tester system prompt
2. Aggregate metrics from the latest generation
3. Raw test results from a batch of pipeline runs

Your job is to produce a precise diagnosis of what the tester is doing wrong and what it is \
doing right.

RULES:
- Identify exactly 3 failure patterns. Each must be SPECIFIC and ACTIONABLE.
  GOOD: "The tester misses division-by-zero checks in 70% of arithmetic tasks — no test calls the \
function with a zero divisor."
  BAD: "The tester should be more thorough."
- Strengths must be concrete observations, not generic praise.
- Proposed fixes must be phrased as direct prompt instructions that can be inserted verbatim.
- Rank proposed_fixes by expected impact (highest first).

Respond ONLY with JSON, no preamble, no markdown fences:
{
  "failure_patterns": ["<specific pattern 1>", "<specific pattern 2>", "<specific pattern 3>"],
  "strengths_to_keep": ["<strength 1>", "<strength 2>"],
  "proposed_fixes": ["<fix instruction 1>", "<fix instruction 2>", "<fix instruction 3>"]
}"""


class AnalysisResult(BaseModel):
    """Output from the failure pattern analyzer."""

    failure_patterns: list[str] = Field(
        description="Top 3 specific, actionable failure patterns observed this generation"
    )
    strengths_to_keep: list[str] = Field(
        description="Concrete strengths the evolver must preserve in the next prompt"
    )
    proposed_fixes: list[str] = Field(
        description="Direct prompt instructions ranked by expected impact (highest first)"
    )


def get_llm() -> ChatAnthropic:
    """Return the Sonnet model used for analysis."""
    return ChatAnthropic(
        model=os.getenv("ANALYZER_MODEL", "claude-sonnet-4-6"),
        max_tokens=2048,
    )


def _build_user_message(
    current_prompt: str,
    metrics: GenerationMetrics,
    raw_test_results: list[dict],
) -> str:
    """Format the analyzer's human-turn input."""
    metrics_summary = {
        "generation": metrics.generation,
        "bug_detection_rate": metrics.bug_detection_rate,
        "false_failure_rate": metrics.false_failure_rate,
        "redundancy_rate": metrics.redundancy_rate,
        "coverage_quality": metrics.coverage_quality,
        "edge_case_coverage": metrics.edge_case_coverage,
        "overall_score": metrics.overall_score,
        "strengths": metrics.strengths,
        "weaknesses": metrics.weaknesses,
    }

    parts = [
        "## Current Tester System Prompt\n",
        current_prompt,
        "\n\n## Generation Metrics\n",
        json.dumps(metrics_summary, indent=2),
        "\n\n## Raw Test Results (sample)\n",
        json.dumps(raw_test_results[:10], indent=2),  # cap at 10 to control token usage
    ]
    return "\n".join(parts)


def analyze(
    current_prompt: str,
    metrics: GenerationMetrics,
    raw_test_results: list[dict],
) -> AnalysisResult:
    """Identify failure patterns and strengths from the latest generation's results.

    Uses Claude Sonnet as the analyzer to produce specific, actionable diagnosis:
    failure patterns to fix, strengths to preserve, and ranked prompt-fix instructions.

    Args:
        current_prompt: The tester system prompt used in this generation.
        metrics: Aggregate GenerationMetrics from the evaluator.
        raw_test_results: Per-task test result dicts from the pipeline run.

    Returns:
        AnalysisResult with failure_patterns, strengths_to_keep, and proposed_fixes.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=_ANALYZER_PROMPT),
        HumanMessage(content=_build_user_message(current_prompt, metrics, raw_test_results)),
    ]

    try:
        response = llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else str(response.content)

        # Strip markdown fences if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        data = json.loads(text.strip())
        return AnalysisResult(**data)

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        # Fallback: surface a minimal result so the evolution loop never crashes
        return AnalysisResult(
            failure_patterns=[
                f"Analyzer failed to parse LLM response: {exc}",
                "Unable to determine failure pattern 2",
                "Unable to determine failure pattern 3",
            ],
            strengths_to_keep=metrics.strengths[:2] if metrics.strengths else [],
            proposed_fixes=["Review raw test results manually to identify patterns."],
        )

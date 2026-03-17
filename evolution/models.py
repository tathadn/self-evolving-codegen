from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TestEffectivenessScore(BaseModel):
    """Per-test-case evaluation from the LLM judge.

    Captures whether an individual generated test caught a real bug,
    produced a false failure, was redundant, or covered a useful category.
    """

    test_name: str = Field(description="Name or identifier of the test case")
    caught_real_bug: bool = Field(description="Whether the test detected a genuine defect")
    was_redundant: bool = Field(description="Whether the test duplicates coverage of another test")
    was_false_failure: bool = Field(
        description="Whether the test fails for a reason unrelated to actual code correctness"
    )
    coverage_category: str = Field(
        description="One of: happy_path, edge_case, error_handling, integration"
    )


class GenerationMetrics(BaseModel):
    """Aggregate metrics for one evolution generation.

    Summarises the tester's overall performance across a batch of pipeline
    runs, including per-dimension scores and qualitative observations.
    """

    generation: int = Field(description="Zero-based generation index")
    bug_detection_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of real bugs caught across all tasks"
    )
    false_failure_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of tests that failed for invalid reasons"
    )
    redundancy_rate: float = Field(
        ge=0.0, le=1.0, description="Fraction of tests that duplicate coverage"
    )
    coverage_quality: float = Field(
        ge=1.0, le=10.0, description="LLM-judge score for breadth and depth of coverage"
    )
    edge_case_coverage: float = Field(
        ge=1.0, le=10.0, description="LLM-judge score for edge-case handling"
    )
    overall_score: float = Field(
        ge=0.0, le=1.0, description="Weighted composite score across all dimensions"
    )
    strengths: list[str] = Field(
        default_factory=list, description="Observed strengths to preserve in the next generation"
    )
    weaknesses: list[str] = Field(
        default_factory=list,
        description="Specific, actionable failure patterns to address in the next generation",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EvolutionHistory(BaseModel):
    """Full log of an evolution experiment.

    Tracks every generation's metrics and the corresponding prompt text so
    that the best-performing generation can be identified and replayed.
    """

    experiment_name: str = Field(description="Human-readable identifier for the experiment run")
    generations: list[GenerationMetrics] = Field(
        default_factory=list, description="Ordered list of per-generation metrics"
    )
    prompt_versions: dict[int, str] = Field(
        default_factory=dict,
        description="Mapping from generation number to the tester system prompt used",
    )

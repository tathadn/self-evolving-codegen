"""Mock data for testing evolution components without API calls.

Use this module in all Phase 1-3 tests. It provides realistic but fabricated
GenerationMetrics, AnalysisResult, and raw test result dicts.

Cost: $0.00 — no API calls anywhere in this file.
"""

from __future__ import annotations

from datetime import datetime

from evolution.analyzer import AnalysisResult
from evolution.models import GenerationMetrics, TestEffectivenessScore


# ── Fake per-test scores ───────────────────────────────────────────────────────

def make_test_score(
    test_name: str,
    caught_real_bug: bool = True,
    was_redundant: bool = False,
    was_false_failure: bool = False,
    coverage_category: str = "happy_path",
) -> TestEffectivenessScore:
    """Return a single fake TestEffectivenessScore."""
    return TestEffectivenessScore(
        test_name=test_name,
        caught_real_bug=caught_real_bug,
        was_redundant=was_redundant,
        was_false_failure=was_false_failure,
        coverage_category=coverage_category,
    )


MOCK_TEST_SCORES: list[TestEffectivenessScore] = [
    make_test_score("test_add_positive", caught_real_bug=True, coverage_category="happy_path"),
    make_test_score("test_divide_by_zero", caught_real_bug=True, coverage_category="error_handling"),
    make_test_score("test_subtract_negative", caught_real_bug=False, coverage_category="edge_case"),
    make_test_score("test_multiply_zero", caught_real_bug=True, coverage_category="edge_case"),
    make_test_score(
        "test_add_positive_copy",
        caught_real_bug=False,
        was_redundant=True,
        coverage_category="happy_path",
    ),
    make_test_score(
        "test_import_error",
        caught_real_bug=False,
        was_false_failure=True,
        coverage_category="happy_path",
    ),
]


# ── Fake GenerationMetrics (5 generations showing improvement) ─────────────────

def _make_metrics(
    generation: int,
    bug_detection_rate: float,
    false_failure_rate: float,
    redundancy_rate: float,
    coverage_quality: float,
    edge_case_coverage: float,
    overall_score: float,
    strengths: list[str],
    weaknesses: list[str],
) -> GenerationMetrics:
    return GenerationMetrics(
        generation=generation,
        bug_detection_rate=bug_detection_rate,
        false_failure_rate=false_failure_rate,
        redundancy_rate=redundancy_rate,
        coverage_quality=coverage_quality,
        edge_case_coverage=edge_case_coverage,
        overall_score=overall_score,
        strengths=strengths,
        weaknesses=weaknesses,
        timestamp=datetime(2026, 3, 1, 12, 0, generation),
    )


MOCK_METRICS: list[GenerationMetrics] = [
    _make_metrics(
        generation=0,
        bug_detection_rate=0.40,
        false_failure_rate=0.20,
        redundancy_rate=0.30,
        coverage_quality=4.5,
        edge_case_coverage=3.0,
        overall_score=0.38,
        strengths=["Tests basic happy path well"],
        weaknesses=["Misses division-by-zero in 60% of arithmetic tasks", "High redundancy rate"],
    ),
    _make_metrics(
        generation=1,
        bug_detection_rate=0.52,
        false_failure_rate=0.16,
        redundancy_rate=0.24,
        coverage_quality=5.5,
        edge_case_coverage=4.2,
        overall_score=0.47,
        strengths=["Tests basic happy path well", "Improved error handling coverage"],
        weaknesses=["Still misses boundary conditions for string inputs"],
    ),
    _make_metrics(
        generation=2,
        bug_detection_rate=0.63,
        false_failure_rate=0.12,
        redundancy_rate=0.18,
        coverage_quality=6.5,
        edge_case_coverage=5.8,
        overall_score=0.56,
        strengths=["Good error handling coverage", "Low false failure rate"],
        weaknesses=["Integration tests missing for multi-method flows"],
    ),
    _make_metrics(
        generation=3,
        bug_detection_rate=0.71,
        false_failure_rate=0.08,
        redundancy_rate=0.12,
        coverage_quality=7.2,
        edge_case_coverage=6.9,
        overall_score=0.65,
        strengths=["Strong edge case detection", "Low redundancy"],
        weaknesses=["Weak integration test coverage"],
    ),
    _make_metrics(
        generation=4,
        bug_detection_rate=0.80,
        false_failure_rate=0.06,
        redundancy_rate=0.08,
        coverage_quality=8.1,
        edge_case_coverage=7.8,
        overall_score=0.74,
        strengths=["High bug detection rate", "Minimal redundancy", "Good edge case coverage"],
        weaknesses=["Some tests still too tightly coupled to implementation"],
    ),
]


# ── Fake AnalysisResult ────────────────────────────────────────────────────────

MOCK_ANALYSIS: AnalysisResult = AnalysisResult(
    failure_patterns=[
        "The tester misses division-by-zero checks in 70% of arithmetic tasks — "
        "no test calls the function with a zero divisor.",
        "Tests for string inputs never check empty string ('') — affects 80% of tasks "
        "that accept str parameters.",
        "Redundant tests duplicate the happy path with only cosmetically different inputs "
        "(e.g. add(1,2) and add(2,3)) in 40% of generated test files.",
    ],
    strengths_to_keep=[
        "Consistently generates import statements correctly — no import errors observed.",
        "Always tests the primary happy path for each public method.",
    ],
    proposed_fixes=[
        "After generating tests, explicitly add one test per function that passes a zero, "
        "empty string, or None as each parameter.",
        "Before finalising, scan the test list and remove any test whose assertion is "
        "equivalent to an existing test with different literal values.",
        "Add at least one test that chains two or more method calls to verify state is "
        "correctly maintained across invocations.",
    ],
)


# ── Fake raw test result dicts (as produced by the pipeline) ───────────────────

MOCK_RAW_RESULTS: list[dict] = [
    {
        "task": "A Python calculator with add, subtract, multiply, divide",
        "passed": True,
        "total_tests": 6,
        "passed_tests": 5,
        "errors": ["FAILED test_divide_by_zero - ZeroDivisionError not raised"],
        "per_test_results": [
            {"name": "test_add", "passed": True},
            {"name": "test_subtract", "passed": True},
            {"name": "test_multiply", "passed": True},
            {"name": "test_divide", "passed": True},
            {"name": "test_divide_by_zero", "passed": False},
            {"name": "test_add_floats", "passed": True},
        ],
    },
    {
        "task": "A Python linked list with insert, delete, search, reverse",
        "passed": True,
        "total_tests": 8,
        "passed_tests": 8,
        "errors": [],
        "per_test_results": [
            {"name": f"test_{op}", "passed": True}
            for op in ["insert", "delete", "search", "reverse", "empty", "single", "duplicate", "order"]
        ],
    },
    {
        "task": "A Python password validator",
        "passed": False,
        "total_tests": 5,
        "passed_tests": 3,
        "errors": [
            "FAILED test_empty_password - AssertionError: expected False, got True",
            "FAILED test_special_chars_only - AssertionError: expected False, got True",
        ],
        "per_test_results": [
            {"name": "test_valid_password", "passed": True},
            {"name": "test_too_short", "passed": True},
            {"name": "test_no_uppercase", "passed": True},
            {"name": "test_empty_password", "passed": False},
            {"name": "test_special_chars_only", "passed": False},
        ],
    },
]

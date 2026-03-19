"""Unit tests for evolution/evaluator.py — tests pure logic only, $0 API cost."""

from __future__ import annotations

from evolution.evaluator import _aggregate, _compute_overall_score
from evolution.mock_data import MOCK_TEST_SCORES


class TestComputeOverallScore:
    def test_perfect_score(self):
        score = _compute_overall_score(
            bug_detection_rate=1.0,
            false_failure_rate=0.0,
            redundancy_rate=0.0,
            coverage_quality=10.0,
            edge_case_coverage=10.0,
        )
        assert score == 1.0

    def test_zero_score(self):
        score = _compute_overall_score(
            bug_detection_rate=0.0,
            false_failure_rate=1.0,
            redundancy_rate=1.0,
            coverage_quality=1.0,
            edge_case_coverage=1.0,
        )
        assert score == 0.0

    def test_midpoint_score(self):
        score = _compute_overall_score(
            bug_detection_rate=0.5,
            false_failure_rate=0.5,
            redundancy_rate=0.5,
            coverage_quality=5.5,
            edge_case_coverage=5.5,
        )
        assert 0.0 <= score <= 1.0

    def test_inverted_metrics(self):
        # Higher false_failure_rate should lower the score
        score_low_ffr = _compute_overall_score(0.8, 0.0, 0.0, 8.0, 8.0)
        score_high_ffr = _compute_overall_score(0.8, 1.0, 0.0, 8.0, 8.0)
        assert score_low_ffr > score_high_ffr

    def test_output_clamped(self):
        # Should never exceed [0, 1]
        score = _compute_overall_score(1.0, 0.0, 0.0, 10.0, 10.0)
        assert 0.0 <= score <= 1.0


class TestAggregate:
    def test_aggregate_with_mock_scores(self):
        metrics = _aggregate(
            scores=MOCK_TEST_SCORES,
            coverage_quality=6.5,
            edge_case_coverage=5.0,
            strengths=["Tests basic path"],
            weaknesses=["Misses empty inputs"],
            generation=0,
        )
        assert metrics.generation == 0
        assert 0.0 <= metrics.bug_detection_rate <= 1.0
        assert 0.0 <= metrics.false_failure_rate <= 1.0
        assert 0.0 <= metrics.redundancy_rate <= 1.0
        assert metrics.coverage_quality == 6.5
        assert metrics.strengths == ["Tests basic path"]

    def test_aggregate_empty_scores(self):
        metrics = _aggregate(
            scores=[],
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            strengths=[],
            weaknesses=[],
            generation=1,
        )
        assert metrics.bug_detection_rate == 0.0
        assert metrics.false_failure_rate == 0.0
        assert metrics.redundancy_rate == 0.0

    def test_aggregate_rates_match_counts(self):
        """Verify rates computed correctly from the 6 mock scores."""
        # MOCK_TEST_SCORES: 3 caught_real_bug, 1 redundant, 1 false_failure
        metrics = _aggregate(
            scores=MOCK_TEST_SCORES,
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            strengths=[],
            weaknesses=[],
            generation=0,
        )
        assert abs(metrics.bug_detection_rate - 3 / 6) < 0.001
        assert abs(metrics.false_failure_rate - 1 / 6) < 0.001
        assert abs(metrics.redundancy_rate - 1 / 6) < 0.001

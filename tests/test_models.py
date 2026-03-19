"""Unit tests for evolution/models.py — $0 API cost."""

from __future__ import annotations

from datetime import datetime

import pytest

from evolution.models import EvolutionHistory, GenerationMetrics, TestEffectivenessScore


class TestTestEffectivenessScore:
    def test_defaults(self):
        score = TestEffectivenessScore(test_name="test_foo")
        assert score.caught_real_bug is False
        assert score.was_redundant is False
        assert score.was_false_failure is False
        assert score.coverage_category == "happy_path"

    def test_all_fields(self):
        score = TestEffectivenessScore(
            test_name="test_divide_by_zero",
            caught_real_bug=True,
            was_redundant=False,
            was_false_failure=False,
            coverage_category="error_handling",
        )
        assert score.test_name == "test_divide_by_zero"
        assert score.coverage_category == "error_handling"


class TestGenerationMetrics:
    def test_valid_metrics(self):
        m = GenerationMetrics(
            generation=0,
            bug_detection_rate=0.75,
            false_failure_rate=0.10,
            redundancy_rate=0.05,
            coverage_quality=7.5,
            edge_case_coverage=6.0,
            overall_score=0.65,
        )
        assert m.generation == 0
        assert m.overall_score == 0.65
        assert isinstance(m.timestamp, datetime)

    def test_default_lists(self):
        m = GenerationMetrics(
            generation=1,
            bug_detection_rate=0.5,
            false_failure_rate=0.1,
            redundancy_rate=0.1,
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            overall_score=0.5,
        )
        assert m.strengths == []
        assert m.weaknesses == []

    def test_rate_bounds_enforced(self):
        with pytest.raises(Exception):
            GenerationMetrics(
                generation=0,
                bug_detection_rate=1.5,  # > 1.0 — invalid
                false_failure_rate=0.0,
                redundancy_rate=0.0,
                coverage_quality=5.0,
                edge_case_coverage=5.0,
                overall_score=0.5,
            )

    def test_serialization_roundtrip(self):
        m = GenerationMetrics(
            generation=2,
            bug_detection_rate=0.8,
            false_failure_rate=0.05,
            redundancy_rate=0.10,
            coverage_quality=8.0,
            edge_case_coverage=7.0,
            overall_score=0.70,
            strengths=["good edge cases"],
            weaknesses=["misses empty input"],
        )
        data = m.model_dump()
        restored = GenerationMetrics(**data)
        assert restored.generation == m.generation
        assert restored.overall_score == m.overall_score
        assert restored.strengths == m.strengths


class TestEvolutionHistory:
    def test_empty_history(self):
        h = EvolutionHistory(experiment_name="test_exp")
        assert h.experiment_name == "test_exp"
        assert h.generations == []
        assert h.prompt_versions == {}

    def test_add_generation(self):
        h = EvolutionHistory(experiment_name="test_exp")
        m = GenerationMetrics(
            generation=0,
            bug_detection_rate=0.5,
            false_failure_rate=0.1,
            redundancy_rate=0.1,
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            overall_score=0.5,
        )
        h.generations.append(m)
        h.prompt_versions[0] = "You are a tester."
        assert len(h.generations) == 1
        assert h.prompt_versions[0] == "You are a tester."

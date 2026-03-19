"""Unit tests for evolution/tracker.py — $0 API cost."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from evolution.mock_data import MOCK_ANALYSIS, MOCK_METRICS
from evolution.tracker import EvolutionTracker, estimate_generation_cost, estimate_total_cost


@pytest.fixture
def tracker(tmp_path: Path) -> EvolutionTracker:
    """Return a tracker pointing at a temp directory."""
    with patch("evolution.tracker._EXPERIMENTS_DIR", tmp_path):
        return EvolutionTracker("test_exp")


class TestEvolutionTracker:
    def test_creates_experiment_dir(self, tmp_path: Path):
        with patch("evolution.tracker._EXPERIMENTS_DIR", tmp_path):
            tracker = EvolutionTracker("my_exp")
        assert (tmp_path / "my_exp").is_dir()

    def test_log_and_retrieve_generation(self, tracker: EvolutionTracker):
        metrics = MOCK_METRICS[0]
        tracker.log_generation(metrics, "You are a tester.", MOCK_ANALYSIS)

        history = tracker.get_performance_history()
        assert len(history) == 1
        assert history[0].generation == 0

    def test_get_best_generation(self, tracker: EvolutionTracker):
        for m in MOCK_METRICS:
            tracker.log_generation(m, f"Prompt gen {m.generation}")

        best = tracker.get_best_generation()
        assert best is not None
        # Generation 4 has the highest overall_score in mock data
        assert best.generation == 4

    def test_get_prompt(self, tracker: EvolutionTracker):
        tracker.log_generation(MOCK_METRICS[0], "initial prompt text")
        retrieved = tracker.get_prompt(0)
        assert retrieved == "initial prompt text"

    def test_get_prompt_missing_returns_none(self, tracker: EvolutionTracker):
        assert tracker.get_prompt(99) is None

    def test_metrics_file_written(self, tracker: EvolutionTracker):
        metrics = MOCK_METRICS[1]
        tracker.log_generation(metrics, "some prompt")
        metrics_file = tracker.experiment_dir / "metrics_gen_1.json"
        assert metrics_file.exists()
        data = json.loads(metrics_file.read_text())
        assert data["generation"] == 1

    def test_prompt_file_written(self, tracker: EvolutionTracker):
        tracker.log_generation(MOCK_METRICS[0], "my prompt text")
        prompt_file = tracker.experiment_dir / "prompt_gen_0.txt"
        assert prompt_file.exists()
        assert prompt_file.read_text() == "my prompt text"

    def test_analysis_file_written(self, tracker: EvolutionTracker):
        tracker.log_generation(MOCK_METRICS[0], "prompt", MOCK_ANALYSIS)
        analysis_file = tracker.experiment_dir / "analysis_gen_0.json"
        assert analysis_file.exists()
        data = json.loads(analysis_file.read_text())
        assert "failure_patterns" in data
        assert len(data["failure_patterns"]) == 3

    def test_history_persists_across_instances(self, tmp_path: Path):
        """Verify history survives a new EvolutionTracker instance (disk round-trip)."""
        with patch("evolution.tracker._EXPERIMENTS_DIR", tmp_path):
            t1 = EvolutionTracker("persist_test")
            t1.log_generation(MOCK_METRICS[0], "prompt v0")

            t2 = EvolutionTracker("persist_test")
            history = t2.get_performance_history()

        assert len(history) == 1
        assert history[0].generation == 0

    def test_log_generation_deduplicates(self, tracker: EvolutionTracker):
        """Re-logging the same generation replaces it, not appends."""
        tracker.log_generation(MOCK_METRICS[0], "first prompt")
        tracker.log_generation(MOCK_METRICS[0], "second prompt")

        history = tracker.get_performance_history()
        assert len(history) == 1
        assert tracker.get_prompt(0) == "second prompt"

    def test_empty_history_returns_none_for_best(self, tracker: EvolutionTracker):
        assert tracker.get_best_generation() is None

    def test_five_generations_sorted(self, tracker: EvolutionTracker):
        for m in reversed(MOCK_METRICS):
            tracker.log_generation(m, f"prompt gen {m.generation}")

        history = tracker.get_performance_history()
        gens = [m.generation for m in history]
        assert gens == sorted(gens)


class TestCostEstimation:
    def test_estimate_generation_cost_sonnet(self):
        cost = estimate_generation_cost(batch_size=5, use_opus=False)
        assert cost > 0
        # 5 tasks × (sonnet×5) + 5×haiku + sonnet×2
        expected = 5 * (0.06 * 5) + 5 * 0.005 + 0.06 * 2
        assert abs(cost - round(expected, 3)) < 0.001

    def test_estimate_generation_cost_opus(self):
        cost_sonnet = estimate_generation_cost(batch_size=5, use_opus=False)
        cost_opus = estimate_generation_cost(batch_size=5, use_opus=True)
        assert cost_opus > cost_sonnet

    def test_estimate_total_cost(self):
        total = estimate_total_cost(generations=5, batch_size=3)
        per_gen = estimate_generation_cost(batch_size=3)
        assert abs(total - round(5 * per_gen, 2)) < 0.01

    def test_total_cost_scales_with_generations(self):
        cost_2 = estimate_total_cost(generations=2, batch_size=3)
        cost_4 = estimate_total_cost(generations=4, batch_size=3)
        assert abs(cost_4 - 2 * cost_2) < 0.01

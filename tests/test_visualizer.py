"""Unit tests for evolution/visualize.py — $0 API cost."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolution.mock_data import MOCK_METRICS
from evolution.visualize import plot_evolution


class TestPlotEvolution:
    def test_produces_png_file(self, tmp_path: Path):
        out = tmp_path / "test_exp" / "evolution_chart.png"
        chart_path = plot_evolution(MOCK_METRICS, "test_exp", save_path=out)
        assert chart_path.exists()
        assert chart_path.suffix == ".png"
        assert chart_path.stat().st_size > 0

    def test_returns_correct_path(self, tmp_path: Path):
        out = tmp_path / "my_exp" / "evolution_chart.png"
        chart_path = plot_evolution(MOCK_METRICS, "my_exp", save_path=out)
        assert chart_path == out

    def test_single_generation(self, tmp_path: Path):
        """Visualizer should not crash with only one data point."""
        out = tmp_path / "single" / "evolution_chart.png"
        chart_path = plot_evolution(MOCK_METRICS[:1], "single", save_path=out)
        assert chart_path.exists()

    def test_creates_parent_directory(self, tmp_path: Path):
        """Parent dir should be created automatically."""
        out = tmp_path / "nested" / "deep" / "evolution_chart.png"
        chart_path = plot_evolution(MOCK_METRICS, "exp", save_path=out)
        assert chart_path.exists()

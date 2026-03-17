from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from evolution.models import GenerationMetrics

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


def plot_evolution(
    history: list[GenerationMetrics],
    experiment_name: str,
    save_path: Path | None = None,
) -> Path:
    """Generate and save a multi-panel evolution performance chart.

    Creates four subplots:
    1. Overall score across generations
    2. Per-dimension breakdown (bug detection, false failure, etc.)
    3. Score delta from generation 0
    4. Strength / weakness observation counts

    Args:
        history: Ordered list of GenerationMetrics from all logged generations.
        experiment_name: Used as the chart title and default save directory.
        save_path: Where to save the PNG. Defaults to
            ``experiments/{experiment_name}/evolution_chart.png``.

    Returns:
        Path where the chart was saved.
    """
    if save_path is None:
        save_path = _EXPERIMENTS_DIR / experiment_name / "evolution_chart.png"

    gens = [m.generation for m in history]
    overall = [m.overall_score for m in history]
    bug_detection = [m.bug_detection_rate for m in history]
    false_failure = [m.false_failure_rate for m in history]
    redundancy = [m.redundancy_rate for m in history]
    coverage_q = [(m.coverage_quality - 1) / 9 for m in history]  # normalised 0-1
    edge_case = [(m.edge_case_coverage - 1) / 9 for m in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Evolution Performance: {experiment_name}", fontsize=14, fontweight="bold")

    # ── Panel 1: Overall score ───────────────────────────────────────────────
    ax1 = axes[0, 0]
    ax1.plot(gens, overall, "b-o", linewidth=2, markersize=6, label="Overall score")
    ax1.fill_between(gens, overall, alpha=0.15, color="blue")
    ax1.set_title("Overall Score", fontweight="bold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Score (0–1)")
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.grid(True, alpha=0.3)
    if len(gens) > 1:
        best_idx = overall.index(max(overall))
        best_gen = gens[best_idx]
        ax1.axvline(
            x=best_gen, color="green", linestyle="--", alpha=0.5, label=f"Best: gen {best_gen}"
        )
    ax1.legend(fontsize=8)

    # ── Panel 2: Per-dimension breakdown ────────────────────────────────────
    ax2 = axes[0, 1]
    x = np.arange(len(gens))
    width = 0.15
    ax2.bar(x - 2 * width, bug_detection, width, label="Bug detection", color="steelblue")
    ax2.bar(
        x - width,
        [1 - v for v in false_failure],
        width,
        label="1 − False failure",
        color="seagreen",
    )
    ax2.bar(x, coverage_q, width, label="Coverage quality", color="darkorange")
    ax2.bar(x + width, edge_case, width, label="Edge case", color="mediumpurple")
    ax2.bar(
        x + 2 * width,
        [1 - v for v in redundancy],
        width,
        label="1 − Redundancy",
        color="crimson",
    )
    ax2.set_title("Per-Dimension Breakdown", fontweight="bold")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Score (0–1)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(g) for g in gens])
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: Score delta from generation 0 ──────────────────────────────
    ax3 = axes[1, 0]
    if overall:
        baseline = overall[0]
        deltas = [s - baseline for s in overall]
        colors = ["seagreen" if d >= 0 else "crimson" for d in deltas]
        ax3.bar(gens, deltas, color=colors, alpha=0.8)
        ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_title("Score Delta from Generation 0", fontweight="bold")
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Δ Score")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax3.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Strength / weakness observation counts ─────────────────────
    ax4 = axes[1, 1]
    strength_counts = [len(m.strengths) for m in history]
    weakness_counts = [len(m.weaknesses) for m in history]
    ax4.plot(gens, strength_counts, "g-s", label="Strengths", linewidth=1.5, markersize=5)
    ax4.plot(gens, weakness_counts, "r-^", label="Weaknesses", linewidth=1.5, markersize=5)
    ax4.set_title("Strength / Weakness Observations", fontweight="bold")
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Count")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path

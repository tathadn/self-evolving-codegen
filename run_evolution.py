#!/usr/bin/env python3
"""Main evolution loop for the self-evolving tester.

Usage:
    python run_evolution.py
    python run_evolution.py --generations 3 --batch-size 3
    python run_evolution.py --experiment my_run_001
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, START, StateGraph  # noqa: E402

from agents import (  # noqa: E402
    coder_node,
    orchestrator_node,
    planner_node,
    reviewer_node,
    should_continue,
)
from agents.tester import make_tester_node  # noqa: E402
from evolution import analyzer, evaluator, evolver  # noqa: E402
from evolution.models import GenerationMetrics  # noqa: E402
from evolution.tracker import EvolutionTracker  # noqa: E402
from evolution.visualize import plot_evolution  # noqa: E402
from models.schemas import AgentState, CodeArtifact  # noqa: E402

_PROMPTS_DIR = Path(__file__).parent / "prompts"

SAMPLE_TASKS: list[str] = [
    "A Python calculator that supports add, subtract, multiply, divide with error handling"
    " for division by zero",
    "A Python FastAPI server with /health and /echo POST endpoints",
    "A Python linked list implementation with insert, delete, search, and reverse methods",
    "A Python file-based todo list manager with add, remove, list, and mark-complete operations",
    "A Python password validator that checks length, uppercase, lowercase, digits, and special"
    " characters",
    "A Python CSV parser that reads a file and computes column statistics (mean, median, min, max)",
    "A Python rate limiter class using the token bucket algorithm",
    "A Python LRU cache implementation with get and put operations",
    "A Python Markdown-to-HTML converter supporting headers, bold, italic, and links",
    "A Python binary search tree with insert, delete, search, and in-order traversal",
]


# ── Pipeline helpers ─────────────────────────────────────────────────────────


def _build_graph_for_generation(generation: int):
    """Build a compiled LangGraph pipeline with the tester at the given generation.

    The tester node is the only agent parameterised by generation; all others
    are the standard V1 nodes.

    Args:
        generation: Which evolved prompt version the tester should use.

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("tester", make_tester_node(generation=generation))

    graph.add_edge(START, "orchestrator")
    graph.add_edge("orchestrator", "planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "reviewer")
    graph.add_edge("reviewer", "tester")
    graph.add_conditional_edges("tester", should_continue, {"end": END, "revise": "coder"})

    return graph.compile()


def _run_task(task: str, generation: int, max_iterations: int = 2) -> AgentState:
    """Run one coding task through the pipeline and return the final AgentState.

    Args:
        task: The user coding request to pass to the pipeline.
        generation: Tester prompt generation to use.
        max_iterations: Max coder/reviewer/tester revision cycles.

    Returns:
        The final AgentState after the pipeline completes or exhausts revisions.
    """
    app = _build_graph_for_generation(generation)
    initial = AgentState(user_request=task, max_iterations=max_iterations)
    result = app.invoke(initial, config={"recursion_limit": 50})
    return AgentState(**result)


def _load_current_prompt(generation: int) -> str:
    """Load the tester prompt text for the given generation from disk.

    Tries ``tester_gen_{generation}.txt`` first, then falls back to
    ``tester.md`` (the original V1 prompt).

    Args:
        generation: Generation index to load.

    Returns:
        Prompt text string, or empty string if no file is found.
    """
    gen_path = _PROMPTS_DIR / f"tester_gen_{generation}.txt"
    if gen_path.exists():
        return gen_path.read_text()
    base = _PROMPTS_DIR / "tester.md"
    if base.exists():
        return base.read_text()
    return ""


# ── Metrics aggregation ──────────────────────────────────────────────────────


def _aggregate_batch_metrics(
    per_task_metrics: list[GenerationMetrics], generation: int
) -> GenerationMetrics:
    """Average per-task GenerationMetrics into a single aggregate for the generation.

    Numeric fields are simple averages. overall_score is recomputed from the
    averaged dimensions using the canonical weights. Qualitative lists (strengths,
    weaknesses) are deduplicated and capped at five entries each.

    Args:
        per_task_metrics: Metrics objects from each task in the batch.
        generation: Generation index to stamp on the result.

    Returns:
        A single GenerationMetrics representing the whole generation's performance.
    """
    if not per_task_metrics:
        return GenerationMetrics(
            generation=generation,
            bug_detection_rate=0.0,
            false_failure_rate=0.0,
            redundancy_rate=0.0,
            coverage_quality=5.0,
            edge_case_coverage=5.0,
            overall_score=0.0,
            strengths=[],
            weaknesses=["No tasks were successfully evaluated this generation."],
        )

    n = len(per_task_metrics)
    bug_detection_rate = sum(m.bug_detection_rate for m in per_task_metrics) / n
    false_failure_rate = sum(m.false_failure_rate for m in per_task_metrics) / n
    redundancy_rate = sum(m.redundancy_rate for m in per_task_metrics) / n
    coverage_quality = sum(m.coverage_quality for m in per_task_metrics) / n
    edge_case_coverage = sum(m.edge_case_coverage for m in per_task_metrics) / n

    # Recompute overall score from averaged dimensions
    coverage_quality_norm = (coverage_quality - 1) / 9
    edge_case_norm = (edge_case_coverage - 1) / 9
    overall_score = round(
        0.30 * bug_detection_rate
        + 0.25 * (1 - false_failure_rate)
        + 0.20 * coverage_quality_norm
        + 0.15 * edge_case_norm
        + 0.10 * (1 - redundancy_rate),
        4,
    )

    # Deduplicated union of qualitative observations, capped at 5 each
    all_strengths = list(dict.fromkeys(s for m in per_task_metrics for s in m.strengths))[:5]
    all_weaknesses = list(dict.fromkeys(w for m in per_task_metrics for w in m.weaknesses))[:5]

    return GenerationMetrics(
        generation=generation,
        bug_detection_rate=round(bug_detection_rate, 4),
        false_failure_rate=round(false_failure_rate, 4),
        redundancy_rate=round(redundancy_rate, 4),
        coverage_quality=round(coverage_quality, 4),
        edge_case_coverage=round(edge_case_coverage, 4),
        overall_score=min(max(overall_score, 0.0), 1.0),
        strengths=all_strengths,
        weaknesses=all_weaknesses,
    )


# ── Main evolution loop ──────────────────────────────────────────────────────


def run_evolution(
    generations: int = 10,
    batch_size: int = 5,
    experiment_name: str = "default",
    max_pipeline_iterations: int = 2,
    rollback_threshold: float = 0.15,
) -> None:
    """Run the full self-evolution experiment.

    For each generation:
    1. Runs ``batch_size`` coding tasks through the pipeline using the tester
       at the current generation.
    2. Evaluates each task's test output with the LLM-as-Judge evaluator.
    3. Aggregates per-task metrics into a single GenerationMetrics.
    4. Logs the generation (metrics, prompt, analysis) via the tracker.
    5. Analyzes failure patterns and strengths (skipped on the final generation).
    6. Evolves the tester prompt (skipped on the final generation).
    7. Rolls back if the new prompt would regress overall_score by more than
       ``rollback_threshold``.

    After all generations, generates the performance chart.

    Args:
        generations: Total number of evolution generations to run.
        batch_size: Number of tasks per generation (drawn cyclically from SAMPLE_TASKS).
        experiment_name: Identifier used as the ``experiments/`` subdirectory name.
        max_pipeline_iterations: Max coder/reviewer/tester revision cycles per task.
        rollback_threshold: Fractional drop in overall_score triggering a rollback.
    """
    print(f"\n{'=' * 60}")
    print(f"  Self-Evolving Tester — Experiment: {experiment_name}")
    print(f"  Generations: {generations}  |  Batch size: {batch_size}")
    print(f"{'=' * 60}\n")

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    tracker = EvolutionTracker(experiment_name)

    # Repeat SAMPLE_TASKS cyclically if batch_size > len(SAMPLE_TASKS)
    tasks = (SAMPLE_TASKS * ((batch_size // len(SAMPLE_TASKS)) + 1))[:batch_size]

    prev_overall: float = 0.0

    for gen in range(generations):
        print(f"\n[Gen {gen}] Starting batch of {len(tasks)} task(s)...")

        per_task_metrics: list[GenerationMetrics] = []
        raw_results: list[dict] = []

        for i, task in enumerate(tasks, 1):
            print(f"  [{i}/{len(tasks)}] {task[:72]}...")
            try:
                state = _run_task(task, generation=gen, max_iterations=max_pipeline_iterations)

                # Build synthetic test artifact for the evaluator if test_code is available
                test_artifacts: list[CodeArtifact] = []
                if state.test_result and state.test_result.test_code:
                    test_artifacts = [
                        CodeArtifact(
                            filename="test_generated.py",
                            language="python",
                            content=state.test_result.test_code,
                        )
                    ]

                if state.test_result is not None:
                    task_metrics = evaluator.evaluate(
                        generation=gen,
                        user_request=task,
                        artifacts=state.artifacts,
                        test_artifacts=test_artifacts,
                        test_result=state.test_result,
                    )
                    per_task_metrics.append(task_metrics)
                    raw_results.append(
                        {
                            "task": task,
                            "passed": state.test_result.passed,
                            "total_tests": state.test_result.total_tests,
                            "passed_tests": state.test_result.passed_tests,
                            "errors": state.test_result.errors[:3],
                            "per_test_results": state.test_result.per_test_results or [],
                        }
                    )
                else:
                    print(f"    WARNING: No test result for task {i} — skipping evaluation.")

            except Exception as exc:
                print(f"    ERROR on task {i}: {exc}")

        if not per_task_metrics:
            print(f"[Gen {gen}] No tasks completed — skipping generation.")
            continue

        metrics = _aggregate_batch_metrics(per_task_metrics, gen)
        prompt_text = _load_current_prompt(gen)

        print(
            f"[Gen {gen}] Overall: {metrics.overall_score:.3f} "
            f"| Bug detection: {metrics.bug_detection_rate:.1%} "
            f"| False failure: {metrics.false_failure_rate:.1%} "
            f"| Tasks evaluated: {len(per_task_metrics)}"
        )

        # Rollback: if gen N performs >rollback_threshold worse than gen N-1, revert prompt
        if (
            gen > 0
            and prev_overall > 0
            and metrics.overall_score < prev_overall * (1 - rollback_threshold)
        ):
            print(
                f"  [Rollback] Score dropped from {prev_overall:.3f} to "
                f"{metrics.overall_score:.3f} (>{rollback_threshold:.0%} regression). "
                f"Reverting gen {gen} prompt to gen {gen - 1}."
            )
            reverted_prompt = tracker.get_prompt(gen - 1) or _load_current_prompt(gen - 1)
            (_PROMPTS_DIR / f"tester_gen_{gen}.txt").write_text(reverted_prompt)
            prompt_text = reverted_prompt

        # Analyze (not on the final generation — no next gen to evolve into)
        analysis = None
        if gen < generations - 1:
            print(f"[Gen {gen}] Analyzing failure patterns...")
            try:
                analysis = analyzer.analyze(prompt_text, metrics, raw_results)
                print(f"  Top failure pattern: {analysis.failure_patterns[0]}")
            except Exception as exc:
                print(f"  WARNING: Analyzer failed — {exc}")

        tracker.log_generation(metrics, prompt_text, analysis)
        prev_overall = metrics.overall_score

        # Evolve prompt for the next generation
        if gen < generations - 1 and analysis is not None:
            print(f"[Gen {gen}] Evolving tester prompt → gen {gen + 1}...")
            try:
                _, new_path = evolver.evolve(metrics, analysis, current_prompt=prompt_text)
                print(f"  Saved to {new_path}")
            except Exception as exc:
                print(f"  WARNING: Evolver failed — {exc}")

    # ── Post-loop: visualise and summarise ───────────────────────────────────
    history = tracker.get_performance_history()
    if history:
        try:
            chart_path = plot_evolution(history, experiment_name)
            print(f"\nEvolution chart saved → {chart_path}")
        except Exception as exc:
            print(f"\nWARNING: Could not generate chart — {exc}")

        best = tracker.get_best_generation()
        if best:
            print(f"Best generation: {best.generation} (overall score: {best.overall_score:.3f})")

    print(f"\nExperiment '{experiment_name}' complete.")
    print(f"Results saved in experiments/{experiment_name}/")


# ── CLI entry point ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evolution loop."""
    parser = argparse.ArgumentParser(
        description="Run the self-evolving tester evolution loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Total number of evolution generations to run",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of coding tasks evaluated per generation",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated from timestamp)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exp_name = args.experiment or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_evolution(
        generations=args.generations,
        batch_size=args.batch_size,
        experiment_name=exp_name,
    )

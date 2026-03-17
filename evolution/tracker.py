from __future__ import annotations

import json
from pathlib import Path

from evolution.analyzer import AnalysisResult
from evolution.models import EvolutionHistory, GenerationMetrics

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


class EvolutionTracker:
    """JSON-based persistence for a single evolution experiment.

    Writes per-generation artefacts to ``experiments/{experiment_name}/`` and
    maintains a running ``evolution_history.json`` that accumulates metrics and
    prompt versions across all generations.
    """

    def __init__(self, experiment_name: str) -> None:
        """Initialise the tracker and create the experiment directory.

        Args:
            experiment_name: Human-readable identifier used as the directory name.
        """
        self.experiment_name = experiment_name
        self.experiment_dir = _EXPERIMENTS_DIR / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self._history_path = self.experiment_dir / "evolution_history.json"
        self._history = self._load_history()

    # ── Public API ──────────────────────────────────────────────────────────

    def log_generation(
        self,
        metrics: GenerationMetrics,
        prompt_text: str,
        analysis: AnalysisResult | None = None,
    ) -> None:
        """Persist all artefacts for one generation and update the running history.

        Saves:
        - ``metrics_gen_{N}.json`` — raw GenerationMetrics
        - ``prompt_gen_{N}.txt``   — tester prompt used this generation
        - ``analysis_gen_{N}.json`` — AnalysisResult (if provided)
        - ``evolution_history.json`` — updated running log

        Args:
            metrics: Aggregate metrics produced by the evaluator for generation N.
            prompt_text: The tester system prompt used during generation N.
            analysis: Optional AnalysisResult from the analyzer for generation N.
        """
        gen = metrics.generation

        # Per-generation artefacts
        self._write_json(
            self.experiment_dir / f"metrics_gen_{gen}.json",
            json.loads(metrics.model_dump_json()),
        )
        (self.experiment_dir / f"prompt_gen_{gen}.txt").write_text(prompt_text)

        if analysis is not None:
            self._write_json(
                self.experiment_dir / f"analysis_gen_{gen}.json",
                analysis.model_dump(),
            )

        # Update running history
        # Replace existing entry for this generation if re-running
        self._history.generations = [g for g in self._history.generations if g.generation != gen]
        self._history.generations.append(metrics)
        self._history.generations.sort(key=lambda g: g.generation)
        self._history.prompt_versions[gen] = prompt_text

        self._save_history()

    def get_performance_history(self) -> list[GenerationMetrics]:
        """Return all logged GenerationMetrics ordered by generation number."""
        return list(self._history.generations)

    def get_best_generation(self) -> GenerationMetrics | None:
        """Return the GenerationMetrics with the highest overall_score.

        Returns None if no generations have been logged yet.
        """
        if not self._history.generations:
            return None
        return max(self._history.generations, key=lambda g: g.overall_score)

    def get_prompt(self, generation: int) -> str | None:
        """Return the prompt text logged for a specific generation, or None."""
        return self._history.prompt_versions.get(generation)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_history(self) -> EvolutionHistory:
        """Load the running history from disk, or create a fresh one."""
        if self._history_path.exists():
            try:
                data = json.loads(self._history_path.read_text())
                return EvolutionHistory(**data)
            except (json.JSONDecodeError, ValueError):
                pass  # Corrupted file — start fresh
        return EvolutionHistory(experiment_name=self.experiment_name)

    def _save_history(self) -> None:
        """Serialise and write the running history to disk."""
        self._write_json(
            self._history_path,
            json.loads(self._history.model_dump_json()),
        )

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        """Write a dict to a JSON file with readable indentation."""
        path.write_text(json.dumps(data, indent=2, default=str))

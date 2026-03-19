from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from config import EVOLVER_MODEL, MAX_TOKENS
from evolution.analyzer import AnalysisResult
from evolution.models import GenerationMetrics

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_EVOLVER_PROMPT = """You are an expert prompt engineer specialising in AI test generation systems.

You will receive:
1. The current tester system prompt (generation N)
2. Aggregate performance metrics for generation N
3. A diagnosis: failure patterns, strengths to keep, and proposed fix instructions

Your job is to produce an improved tester system prompt (generation N+1).

RULES:
- Make SURGICAL changes — preserve the structure and wording that works, only modify what is broken.
- Do NOT write a completely new prompt. Edit the existing one.
- Insert the proposed fix instructions as concrete, imperative directives (not vague advice).
- Preserve every strength listed in "strengths_to_keep" verbatim or as an equivalent instruction.
- The output prompt MUST be under 1000 words. Cut redundant phrasing to make room for new rules.
- Do NOT include meta-commentary, version notes, or any preamble in the output.
- Output ONLY the new prompt text — nothing else."""


def get_llm() -> ChatAnthropic:
    """Return the Sonnet model used for prompt evolution."""
    return ChatAnthropic(
        model=EVOLVER_MODEL,
        max_tokens=MAX_TOKENS["evolver"],
    )


def _build_user_message(
    current_prompt: str,
    metrics: GenerationMetrics,
    analysis: AnalysisResult,
) -> str:
    """Format the evolver's human-turn input."""
    parts = [
        f"## Current Tester Prompt (Generation {metrics.generation})\n{current_prompt}",
        "## Performance Metrics",
        (
            f"- Overall score: {metrics.overall_score:.3f}\n"
            f"- Bug detection rate: {metrics.bug_detection_rate:.2%}\n"
            f"- False failure rate: {metrics.false_failure_rate:.2%}\n"
            f"- Redundancy rate: {metrics.redundancy_rate:.2%}\n"
            f"- Coverage quality: {metrics.coverage_quality:.1f}/10\n"
            f"- Edge case coverage: {metrics.edge_case_coverage:.1f}/10"
        ),
        "## Diagnosis",
        "### Failure Patterns (must fix)",
        "\n".join(f"{i + 1}. {p}" for i, p in enumerate(analysis.failure_patterns)),
        "### Strengths to Keep",
        "\n".join(f"- {s}" for s in analysis.strengths_to_keep),
        "### Proposed Fix Instructions (ranked by impact)",
        "\n".join(f"{i + 1}. {f}" for i, f in enumerate(analysis.proposed_fixes)),
    ]
    return "\n\n".join(parts)


def _word_count(text: str) -> int:
    """Return approximate word count of a prompt."""
    return len(text.split())


def evolve(
    metrics: GenerationMetrics,
    analysis: AnalysisResult,
    current_prompt: str | None = None,
) -> tuple[str, Path]:
    """Rewrite the tester prompt based on failure analysis.

    Performs surgical edits to the current generation's prompt, inserting
    fix instructions and preserving identified strengths. Saves the result
    as ``prompts/tester_gen_{N+1}.txt``.

    Args:
        metrics: Aggregate metrics from the generation being evolved.
        analysis: AnalysisResult from the analyzer for the same generation.
        current_prompt: The prompt text to evolve. If None, loads from disk
            (``prompts/tester_gen_{N}.txt``, falling back to ``tester.md``).

    Returns:
        Tuple of (new_prompt_text, path_where_it_was_saved).
    """
    if current_prompt is None:
        current_prompt = _load_prompt(metrics.generation)

    llm = get_llm()
    messages = [
        SystemMessage(content=_EVOLVER_PROMPT),
        HumanMessage(content=_build_user_message(current_prompt, metrics, analysis)),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content
        new_prompt = raw if isinstance(raw, str) else str(raw)
        new_prompt = new_prompt.strip()
    except Exception as exc:
        # Fallback: keep the current prompt so the loop can continue
        new_prompt = current_prompt
        new_prompt += f"\n\n# Evolution note (gen {metrics.generation + 1}): evolver failed — {exc}"

    next_gen = metrics.generation + 1
    out_path = _save_prompt(new_prompt, next_gen)
    return new_prompt, out_path


def _load_prompt(generation: int) -> str:
    """Load the tester prompt for the given generation from disk.

    Tries ``tester_gen_{generation}.txt`` first, then falls back to
    ``tester.md`` (the original V1 prompt).
    """
    gen_path = _PROMPTS_DIR / f"tester_gen_{generation}.txt"
    if gen_path.exists():
        return gen_path.read_text()

    base_path = _PROMPTS_DIR / "tester.md"
    if base_path.exists():
        return base_path.read_text()

    raise FileNotFoundError(
        f"No tester prompt found for generation {generation}. Expected {gen_path} or {base_path}."
    )


def _save_prompt(prompt_text: str, generation: int) -> Path:
    """Write the evolved prompt to ``prompts/tester_gen_{generation}.txt``."""
    out_path = _PROMPTS_DIR / f"tester_gen_{generation}.txt"
    out_path.write_text(prompt_text)
    return out_path

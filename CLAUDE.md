# Self-Evolving Code Generator

> V2 of [multi-agent-codegen](https://github.com/tathadn/multi-agent-codegen) — the same pipeline, now with a self-evolving tester.

## Project Overview

A multi-agent AI code generation pipeline (LangGraph + Claude) where the **Tester agent autonomously improves its own test generation strategy** over successive generations through self-evaluation, failure analysis, and prompt evolution.

The five-agent pipeline (Orchestrator → Planner → Coder → Reviewer → Tester) is inherited from V1. V2 adds a **self-evolution engine** that wraps the Tester, observing its performance across batches of pipeline runs and iteratively rewriting its system prompt to produce better tests.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  CODE GENERATION PIPELINE (LangGraph StateGraph)             │
│                                                              │
│  Orchestrator → Planner → Coder → Reviewer → Tester ──┐     │
│                            ▲                           │     │
│                            └── revision loop ──────────┘     │
└──────────────────────────────────┬───────────────────────────┘
                                   │ test results + metadata
                                   ▼
┌──────────────────────────────────────────────────────────────┐
│  SELF-EVOLUTION ENGINE (evolution/)                           │
│                                                              │
│  Evaluator → Analyzer → Evolver → Tracker                    │
│      │                      │                                │
│      │ scores tests         │ writes new prompt              │
│      │ via LLM-as-Judge     │ (prompts/tester_gen_N.txt)     │
│      │                      │                                │
│      └──────────────────────┘                                │
│              ▲                                               │
│              │ next generation uses updated prompt            │
└──────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
self-evolving-codegen/
├── agents/                    # Agent implementations (from V1)
│   ├── __init__.py
│   ├── orchestrator.py        # Parses user intent, sets status
│   ├── planner.py             # Produces structured Plan
│   ├── coder.py               # Generates CodeArtifact objects
│   ├── reviewer.py            # Scores code 0-10, approves/rejects
│   └── tester.py              # ⚡ MODIFIED: supports generation param
├── evolution/                 # 🆕 Self-evolution engine
│   ├── __init__.py
│   ├── models.py              # Pydantic models: TestEffectivenessScore,
│   │                          #   GenerationMetrics, EvolutionHistory
│   ├── evaluator.py           # LLM-as-Judge test scoring
│   ├── analyzer.py            # Failure pattern analysis
│   ├── evolver.py             # Prompt rewriter
│   ├── tracker.py             # JSON persistence + experiment logging
│   └── visualize.py           # Matplotlib performance charts
├── graph/                     # LangGraph workflow definition
│   ├── __init__.py
│   └── workflow.py            # StateGraph with conditional edges
├── models/                    # Pydantic state schemas
│   ├── __init__.py
│   └── state.py               # AgentState, Plan, CodeArtifact,
│                              #   ReviewFeedback, TestResult, TaskStatus
├── prompts/                   # Agent system prompts
│   ├── orchestrator.txt
│   ├── planner.txt
│   ├── coder.txt
│   ├── reviewer.txt
│   ├── tester_gen_0.txt       # Base tester prompt (original)
│   └── tester_gen_N.txt       # 🆕 Evolved versions (auto-generated)
├── sandbox/                   # Docker test runner
│   └── Dockerfile
├── experiments/               # 🆕 Saved evolution run data
│   └── {experiment_name}/
│       ├── evolution_history.json
│       ├── metrics_gen_N.json
│       ├── prompt_gen_N.txt
│       └── evolution_chart.png
├── assets/                    # Screenshots, diagrams
├── app.py                     # Streamlit UI (extended with evolution tab)
├── run_evolution.py           # 🆕 Main evolution loop orchestrator
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── CLAUDE.md                  # This file
```

## Commands

```bash
# ── Setup ──
pip install -e ".[dev]"                   # Install all dependencies
cp .env.example .env                      # Set up environment variables

# ── Run the app ──
streamlit run app.py                      # Launch Streamlit UI at localhost:8501

# ── Run the evolution loop ──
python run_evolution.py                   # Full evolution (default: 10 gens, 5 tasks)
python run_evolution.py --generations 3 --batch-size 3   # Quick test run
python run_evolution.py --experiment my_run_001           # Named experiment

# ── Testing ──
python -m pytest                          # Run project tests
python -m pytest -x -v                    # Verbose, stop on first failure

# ── Linting ──
ruff check .                              # Lint
ruff check . --fix                        # Auto-fix
ruff format .                             # Format

# ── Type checking ──
mypy agents/ models/ graph/ evolution/    # Type check all modules
```

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...              # Anthropic API key

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=self-evolving-codegen
```

## Tech Stack

| Library              | Version  | Purpose                                       |
|----------------------|----------|-----------------------------------------------|
| LangGraph            | ≥ 0.2.0 | Agent orchestration with conditional edges     |
| LangChain Anthropic  | ≥ 0.3.0 | Claude API via `ChatAnthropic`                 |
| LangChain Core       | ≥ 0.3.0 | Message types, structured output               |
| Pydantic             | ≥ 2.0.0 | Typed schemas, structured LLM outputs          |
| Streamlit            | ≥ 1.40.0| Web UI with real-time streaming                |
| Matplotlib           | ≥ 3.8.0 | Evolution performance visualization            |
| python-dotenv        | ≥ 1.0.0 | `.env` file loading                            |
| Docker               | —        | Sandbox for test execution                     |

Python ≥ 3.10 required.

## Coding Conventions

### Style

- **Line length**: 100 characters (ruff configured)
- **Target Python**: 3.10
- **Formatter**: ruff format
- **Linter rules**: E, F, I (isort), UP (pyupgrade)
- Follow PEP 8 conventions

### Type Hints

- Type hints on ALL function signatures — parameters and return types
- Use `Optional[X]` for nullable types, `list[X]` (lowercase) for Python 3.10+
- All Pydantic models must have complete field type annotations

### Docstrings

- Docstrings on all public functions and classes
- Use triple-quote format with a one-line summary, blank line, then details if needed
- Example:
  ```python
  def evaluate_review(agent_review: dict, test_results: list[dict]) -> GenerationMetrics:
      """Score the tester's output using the LLM-as-Judge rubric.

      Assesses each generated test for bug detection, false failure potential,
      redundancy, and edge case coverage. Returns aggregate metrics.
      """
  ```

### Imports

- Standard library first, then third-party, then local — ruff handles sorting
- Use absolute imports for cross-module references: `from models.state import AgentState`
- Never use wildcard imports

### Pydantic Models

- All data flowing between components must be Pydantic `BaseModel` subclasses
- Define models in the appropriate module:
  - Pipeline state models → `models/state.py`
  - Evolution data models → `evolution/models.py`
- Use `model_dump()` for serialization (not the deprecated `.dict()`)
- Provide `Field(description=...)` for complex fields

### Error Handling

- Wrap all LLM API calls in try/except — handle rate limits, parsing failures, timeouts
- When parsing LLM JSON output, always handle markdown code fences:
  ```python
  if "```json" in text:
      text = text.split("```json")[1].split("```")[0]
  ```
- Provide sensible fallback values when JSON parsing fails — never let the pipeline crash
- Log errors with context (which agent, which generation, which PR/task)

### File Organization

- One class per file when the class is complex (agents, evolution components)
- Group related utilities together
- Keep prompt text in `prompts/` directory, not hardcoded in Python files
- Constants (model names, scoring weights) go in a `config.py` or at the top of the relevant module

## Key Data Models

### Existing (from V1 — models/state.py)

```python
class AgentState(BaseModel):
    user_request: str
    plan: Optional[Plan]
    artifacts: list[CodeArtifact]
    review: Optional[ReviewFeedback]
    test_result: Optional[TestResult]
    status: TaskStatus            # PENDING | IN_PROGRESS | COMPLETED | FAILED
    iteration: int
    max_iterations: int

class Plan(BaseModel):
    objective: str
    steps: list[str]
    files_to_create: list[str]
    dependencies: list[str]
    complexity: str               # low | medium | high

class CodeArtifact(BaseModel):
    filename: str
    language: str
    content: str

class ReviewFeedback(BaseModel):
    score: int                    # 0-10
    approved: bool
    issues: list[str]
    suggestions: list[str]

class TestResult(BaseModel):
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    errors: list[str]
    output: str
```

### New (evolution/models.py)

```python
class TestEffectivenessScore(BaseModel):
    """Per-test-case evaluation from the LLM judge."""
    test_name: str
    caught_real_bug: bool
    was_redundant: bool
    was_false_failure: bool
    coverage_category: str        # happy_path | edge_case | error_handling | integration

class GenerationMetrics(BaseModel):
    """Aggregate metrics for one evolution generation."""
    generation: int
    bug_detection_rate: float     # 0.0-1.0
    false_failure_rate: float     # 0.0-1.0
    redundancy_rate: float        # 0.0-1.0
    coverage_quality: float       # 1-10, from LLM judge
    edge_case_coverage: float     # 1-10, from LLM judge
    overall_score: float          # weighted combination
    strengths: list[str]
    weaknesses: list[str]
    timestamp: datetime

class EvolutionHistory(BaseModel):
    """Full log of an evolution experiment."""
    experiment_name: str
    generations: list[GenerationMetrics]
    prompt_versions: dict[int, str]   # generation number -> prompt text
```

## Agent Details

| Agent          | File                    | Model             | Input                              | Output            |
|----------------|-------------------------|-------------------|------------------------------------|-------------------|
| Orchestrator   | `agents/orchestrator.py`| claude-opus-4-6     | user_request                       | Parsed intent     |
| Planner        | `agents/planner.py`     | claude-sonnet-4-6 | user_request                       | Plan              |
| Coder          | `agents/coder.py`       | claude-sonnet-4-6 | Plan + (optional) review/test feedback | list[CodeArtifact]|
| Reviewer       | `agents/reviewer.py`    | claude-sonnet-4-6 | CodeArtifacts                      | ReviewFeedback    |
| Tester         | `agents/tester.py`      | configurable      | CodeArtifacts + Plan               | TestResult        |

### Tester Agent Modifications (V2)

The tester is the ONLY agent modified from V1. Changes are minimal and backward-compatible:

1. **`generation` parameter** (default: 0) — controls which prompt version to load
2. **Prompt loading logic** — `generation=0` loads the original prompt; `generation>0` loads `prompts/tester_gen_{N}.txt`
3. **Extended TestResult metadata** — adds per-test pass/fail breakdown, raw generated test code, and generation number to the output for the evaluator

When `generation=0`, the tester behaves identically to V1. Do NOT modify any other agent.

## Evolution Engine Details

### Evaluator (`evolution/evaluator.py`)
- Uses **Claude Haiku** (cost-efficient) as the LLM-as-Judge
- Structured rubric scoring each test on: bug detection, false failure, redundancy, coverage type
- Returns `GenerationMetrics` with aggregate scores
- Judge prompt stored as a constant at the top of the file

### Analyzer (`evolution/analyzer.py`)
- Receives current prompt + GenerationMetrics + raw test results
- Identifies the top 3 **specific, actionable** failure patterns (not vague advice)
- Outputs: `failure_patterns`, `strengths_to_keep`, `proposed_fixes` (ranked by impact)
- Good pattern: "The tester misses error handling for empty inputs in 70% of tasks"
- Bad pattern: "The tester should be more thorough" (too vague — never produce this)

### Evolver (`evolution/evolver.py`)
- Rewrites the tester's prompt based on the analysis
- **Surgical changes, not complete rewrites** — preserve what works, fix what doesn't
- Adds specific instructions and heuristics, not general advice
- New prompt must stay under 1000 words to avoid prompt bloat
- Saves new prompt as `prompts/tester_gen_{N+1}.txt`

### Tracker (`evolution/tracker.py`)
- JSON-based persistence to `experiments/{experiment_name}/`
- Saves per-generation: metrics, analysis, prompt text
- Running history file: `evolution_history.json`
- Methods: `log_generation()`, `get_performance_history()`, `get_best_generation()`

### Visualizer (`evolution/visualize.py`)
- Matplotlib charts: overall score, per-dimension breakdown, score deltas, prompt length
- Saves to `experiments/{experiment_name}/evolution_chart.png`

## Scoring Weights

```python
WEIGHTS = {
    "bug_detection_rate": 0.30,
    "false_failure_rate": 0.25,   # inverted: lower is better
    "coverage_quality": 0.20,
    "edge_case_coverage": 0.15,
    "redundancy_rate": 0.10,      # inverted: lower is better
}

# overall = sum(weight * score for each metric)
# false_failure and redundancy are inverted: score = 1 - rate
```

## Sample Tasks for Evolution

These are the coding tasks used to evaluate the tester across generations. They range in complexity and domain to test generalization:

```python
SAMPLE_TASKS = [
    "A Python calculator that supports add, subtract, multiply, divide with error handling for division by zero",
    "A Python FastAPI server with /health and /echo POST endpoints",
    "A Python linked list implementation with insert, delete, search, and reverse methods",
    "A Python file-based todo list manager with add, remove, list, and mark-complete operations",
    "A Python password validator that checks length, uppercase, lowercase, digits, and special characters",
    "A Python CSV parser that reads a file and computes column statistics (mean, median, min, max)",
    "A Python rate limiter class using the token bucket algorithm",
    "A Python LRU cache implementation with get and put operations",
    "A Python Markdown-to-HTML converter supporting headers, bold, italic, and links",
    "A Python binary search tree with insert, delete, search, and in-order traversal",
]
```

## Critical Rules

### DO NOT modify these files (inherited from V1, must stay stable):
- `agents/orchestrator.py`
- `agents/planner.py`
- `agents/coder.py`
- `agents/reviewer.py`
- `graph/workflow.py` (unless adding the evolution loop as a separate entry point)
- `models/state.py` (extend with new fields only if backward-compatible)

### The tester modification must be backward-compatible:
- `generation=0` must produce identical behavior to V1
- The `TestResult` model can have new Optional fields but all existing fields must remain unchanged
- The pipeline (`graph/workflow.py`) should work without any knowledge of the evolution system

### Evolution system is a wrapper, not a core change:
- `run_evolution.py` invokes the pipeline programmatically (not through Streamlit)
- Evolution components never import from `graph/` — they only consume pipeline outputs
- The Streamlit app's evolution tab is purely for visualization and comparison, not required for evolution to run

### Cost awareness:
- Use Claude Haiku for the evaluator (judge) to keep costs low
- Use Sonnet for the analyzer and evolver
- Pipeline agents keep their existing model assignments
- Cache aggressively during development — don't re-run pipeline for unchanged generations
- Default batch size of 5 tasks per generation; configurable via CLI

### Git hygiene:
- Commit after each major component (see commit messages below)
- Keep `experiments/` in `.gitignore` except for one sample result
- Never commit `.env` or API keys
- Keep prompt files in version control — they ARE the evolution artifact

## Suggested Commit Flow

```
feat: add evolution data models (evolution/models.py)
feat: add test evaluator with LLM-as-Judge (evolution/evaluator.py)
feat: add failure pattern analyzer (evolution/analyzer.py)
feat: add prompt evolver (evolution/evolver.py)
feat: add evolution tracker with JSON persistence (evolution/tracker.py)
feat: integrate evolution with tester agent (agents/tester.py)
feat: add main evolution loop (run_evolution.py)
feat: add performance visualization (evolution/visualize.py)
feat: add evolution dashboard to Streamlit UI (app.py)
feat: run first evolution experiment + save sample results
docs: write README with architecture, results, and portfolio narrative
chore: cleanup, type hints, docstrings, final lint pass
```

## Debugging Tips

- If the LLM returns malformed JSON, check the prompt — it likely needs a stricter output format instruction with "Respond ONLY with JSON, no preamble"
- If test scores don't improve across generations, check the analyzer output — it may be producing vague recommendations. The analyzer prompt needs to demand specificity.
- If evolution regresses sharply, check the evolver — it may have removed a working strategy. Add rollback logic: if gen N+1 scores >15% worse than gen N, revert to gen N's prompt and try a different evolution path.
- Run `ruff check . --fix && ruff format .` before every commit
- Use `/clear` in Claude Code between major steps to keep context focused

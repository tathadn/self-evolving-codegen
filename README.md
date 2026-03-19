# Self-Evolving Code Generator

> V2 of [multi-agent-codegen](https://github.com/tathadn/multi-agent-codegen) — the same pipeline, now with a self-evolving tester.

A multi-agent AI code generation pipeline (LangGraph + Claude) where the **Tester agent autonomously improves its own test generation strategy** over successive generations through self-evaluation, failure analysis, and prompt evolution.

The five-agent pipeline (Orchestrator → Planner → Coder → Reviewer → Tester) is inherited from V1. V2 adds a **self-evolution engine** that wraps the Tester, observing its performance across batches of pipeline runs and iteratively rewriting its system prompt to produce better tests.

---

## Demo

![Multi-Agent Code Generator — pipeline complete with 79/79 tests passing](assets/demo.png)

*The sidebar shows real-time agent status indicators. A student grade tracker was generated, reviewed (9/10), and passed 79/79 tests on the first attempt.*

---

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

### Code Generation Pipeline

| Agent | Model (Dev) | Model (Final) | Role |
|---|---|---|---|
| **Orchestrator** | claude-sonnet-4-6 | **claude-opus-4-6** | Interprets the request, sets pipeline status |
| **Planner** | claude-sonnet-4-6 | claude-sonnet-4-6 | Produces a structured plan: objective, steps, files, dependencies |
| **Coder** | claude-sonnet-4-6 | claude-sonnet-4-6 | Generates `CodeArtifact` objects; on revisions, receives review issues and test failures as context |
| **Reviewer** | claude-sonnet-4-6 | claude-sonnet-4-6 | Scores code 0–10, flags issues and suggestions |
| **Tester** | claude-sonnet-4-6 | claude-sonnet-4-6 | Generates pytest files for the given prompt generation, runs them in a Docker sandbox |

After the Tester runs, a `should_continue` router either ends the pipeline (`COMPLETED`) or loops back to the Coder if tests failed or the review score is below threshold.

### Self-Evolution Engine

| Component | Model | Role |
|---|---|---|
| **Evaluator** | claude-haiku-4-5 | LLM-as-Judge: scores each test for bug detection, false failure, redundancy, coverage category |
| **Analyzer** | claude-sonnet-4-6 | Identifies the top 3 specific, actionable failure patterns from metrics + raw results |
| **Evolver** | claude-sonnet-4-6 | Makes surgical edits to the tester prompt — preserving strengths, inserting fix instructions |
| **Tracker** | — | JSON persistence to `experiments/{name}/`; cost estimation; maintains running `evolution_history.json` |

#### Scoring Weights

```python
WEIGHTS = {
    "bug_detection_rate": 0.30,   # higher is better
    "false_failure_rate": 0.25,   # inverted: lower is better
    "coverage_quality":   0.20,   # 1-10 normalised to 0-1
    "edge_case_coverage": 0.15,   # 1-10 normalised to 0-1
    "redundancy_rate":    0.10,   # inverted: lower is better
}
```

#### Rollback Logic

If a newly evolved prompt causes the overall score to drop by more than 15% relative to the previous generation, the evolution loop automatically reverts to the prior generation's prompt and continues.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/self-evolving-codegen.git
cd self-evolving-codegen
pip install -e ".[dev]"

# 2. Set your API key
cp .env.example .env
# open .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch the Streamlit UI
streamlit run app.py

# 4. Run the evolution loop (separate terminal)
python run_evolution.py --generations 2 --batch-size 2 --experiment micro_run   # Phase 4: ~$3-5
python run_evolution.py --generations 5 --batch-size 3 --experiment dev_run     # Phase 5: ~$8-12
python run_evolution.py --generations 10 --batch-size 5 --experiment final_run  # Phase 6: ~$18-28

# Phase 6 only: enable Opus for the Orchestrator via env var (do NOT edit config.py)
ORCHESTRATOR_MODEL=claude-opus-4-6 python run_evolution.py --generations 10 --batch-size 5 --experiment final_run
```

---

## Cost Controls

Keeping API spend predictable is a first-class concern. Several mechanisms work together:

### 1. Pipeline result caching (`evolution/cache.py`)

Every pipeline run is cached to `.cache/pipeline_runs/` using a deterministic MD5 key of `{task}:gen{generation}`. Re-running the evolution loop (e.g. after a bug fix) replays all cached results at **$0.00** — only genuinely new task/generation combinations hit the API.

```
[CACHE HIT]  A Python calculator...  (gen 0)   ← free
[API CALL]   A Python linked list... (gen 0)   ← costs money, then cached
```

### 2. Rate limiting (`rate_limited_call`)

All pipeline invocations go through `rate_limited_call()`, which adds a 1.5-second delay between calls to respect Anthropic Pro plan rate limits.

### 3. Cost estimation + confirmation

Before any API spend, the evolution loop prints an estimate and requires explicit confirmation:

```
Estimated cost: $4.23 (cached runs cost $0.00)
Continue? [y/N]
```

### 4. Model tiers

| Role | Model | Why |
|---|---|---|
| Evaluator | **Haiku** (always) | Highest-volume call — ~90% cheaper than Sonnet |
| All other agents | **Sonnet** (dev) / **Opus** (final showcase only) | Quality where it matters |
| Opus override | env var only | `ORCHESTRATOR_MODEL=claude-opus-4-6` — never hardcoded |

### 5. Max token caps

Every LLM call has an explicit `max_tokens` limit set from `config.py` to prevent runaway output costs:

```python
MAX_TOKENS = {
    "orchestrator": 1500, "planner": 2500, "coder": 4000,
    "reviewer": 1500,     "tester": 4000,  "evaluator": 1500,
    "analyzer": 2000,     "evolver": 2000,
}
```

---

## Using the Evolution Loop

### CLI Options

```
--generations N     Total evolution generations (default: 10)
--batch-size N      Coding tasks evaluated per generation (default: 5)
--experiment NAME   Experiment identifier used as the output directory name
```

### What it produces

For each generation, the tracker saves to `experiments/{name}/`:

```
experiments/my_run_001/
├── evolution_history.json   # running log of all GenerationMetrics
├── metrics_gen_0.json
├── metrics_gen_1.json
├── ...
├── prompt_gen_0.txt         # tester prompt used at gen 0
├── prompt_gen_1.txt         # evolved prompt for gen 1
├── ...
├── analysis_gen_0.json      # failure patterns + proposed fixes
└── evolution_chart.png      # 2×2 performance chart
```

### Evolution Dashboard

After running the loop, open the **Evolution Dashboard** tab in the Streamlit app to:

- Browse experiments by name
- View the performance chart
- Compare per-generation metrics in a table
- Diff tester prompts side-by-side
- Read per-generation strengths and weaknesses

---

## Testing (Zero API Cost)

All unit tests run against mock data in `evolution/mock_data.py` — no API calls required.

```bash
python -m pytest tests/ -v           # run all tests
python -m pytest tests/ -x -v       # stop on first failure
```

```
tests/test_models.py      — Pydantic model validation and serialization
tests/test_evaluator.py   — Scoring and aggregation logic (pure functions)
tests/test_tracker.py     — JSON persistence, best-gen selection, cost estimation
tests/test_visualizer.py  — Chart generation from mock data
```

---

## Sample Tasks

The evolution loop evaluates the tester across 10 coding tasks of varying complexity:

- Python calculator with division-by-zero handling
- FastAPI server with `/health` and `/echo` endpoints
- Linked list with insert, delete, search, reverse
- File-based todo list manager
- Password validator (length, case, digits, special chars)
- CSV parser with column statistics
- Token-bucket rate limiter
- LRU cache
- Markdown-to-HTML converter
- Binary search tree

Tasks are drawn cyclically if `--batch-size` exceeds 10.

---

## Key Design Decisions

**Evolution is a wrapper, not a core change.** The pipeline agents (`graph/workflow.py`) are untouched. `run_evolution.py` invokes the pipeline programmatically; evolution components never import from `graph/`. The Tester is the only agent parameterised by generation — `generation=0` produces identical behaviour to V1.

**Surgical prompt edits, not rewrites.** The Evolver is instructed to preserve the structure and wording that works, insert specific imperative directives, and keep the output under 1000 words to prevent prompt bloat.

**Specific failure patterns, not vague advice.** The Analyzer prompt enforces a strict rule: every identified pattern must be concrete and cite observable evidence (e.g. "misses division-by-zero in 70% of arithmetic tasks"), never generic (e.g. "should be more thorough").

**Cost-efficient judging.** The Evaluator uses Claude Haiku to score each test. Sonnet is reserved for the Analyzer and Evolver where reasoning quality matters more. All model assignments are centralised in `config.py` and overrideable via env vars — no model name is hardcoded in agent files.

**Cache-first execution.** Every pipeline run is cached to `.cache/pipeline_runs/`. Re-running the loop (e.g. after fixing a bug in the analyzer) replays all prior results for free. The cache key is deterministic: `MD5("{task}:gen{generation}")`.

---

## Tech Stack

| Library | Version | Purpose |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) | ≥ 0.2.0 | Agent orchestration with conditional edges |
| [LangChain Anthropic](https://github.com/langchain-ai/langchain) | ≥ 0.3.0 | Claude API via `ChatAnthropic` |
| [Pydantic](https://docs.pydantic.dev/) | ≥ 2.0.0 | Typed state schemas and structured LLM outputs |
| [Streamlit](https://streamlit.io/) | ≥ 1.40.0 | Web UI with real-time streaming |
| [Matplotlib](https://matplotlib.org/) | ≥ 3.8.0 | Evolution performance charts |
| [python-dotenv](https://github.com/theskumar/python-dotenv) | ≥ 1.0.0 | `.env` loading |
| Docker | — | Isolated sandbox for test execution |

Python ≥ 3.10 required.

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Model overrides (all default to Sonnet except Evaluator=Haiku — see config.py)
# Uncomment ONLY for the final showcase run (Phase 6):
# ORCHESTRATOR_MODEL=claude-opus-4-6

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=self-evolving-codegen
```

---

## Project Structure

```
self-evolving-codegen/
├── agents/                    # Agent implementations
│   ├── orchestrator.py        # Uses config.ORCHESTRATOR_MODEL
│   ├── planner.py             # Uses config.PLANNER_MODEL
│   ├── coder.py               # Uses config.CODER_MODEL
│   ├── reviewer.py            # Uses config.REVIEWER_MODEL
│   └── tester.py              # Modified: supports generation param
├── evolution/                 # Self-evolution engine
│   ├── models.py              # TestEffectivenessScore, GenerationMetrics, EvolutionHistory
│   ├── evaluator.py           # LLM-as-Judge test scoring (Haiku always)
│   ├── analyzer.py            # Failure pattern analysis
│   ├── evolver.py             # Prompt rewriter
│   ├── tracker.py             # JSON persistence + cost estimation
│   ├── visualize.py           # Matplotlib performance charts
│   ├── cache.py               # Pipeline result caching + rate limiting
│   └── mock_data.py           # Fake data for $0 unit testing
├── graph/
│   └── workflow.py            # LangGraph StateGraph (unchanged from V1)
├── models/
│   └── schemas.py             # AgentState, TestResult (with V2 optional fields)
├── prompts/
│   ├── tester_gen_0.txt       # Base tester prompt
│   └── tester_gen_N.txt       # Auto-generated evolved versions
├── sandbox/
│   └── Dockerfile             # Isolated test runner
├── tests/                     # Unit tests (zero API cost)
│   ├── test_models.py
│   ├── test_evaluator.py
│   ├── test_tracker.py
│   └── test_visualizer.py
├── experiments/               # Saved evolution run data (gitignored)
├── .cache/                    # Cached pipeline results (gitignored)
├── config.py                  # Centralized model names, MAX_TOKENS, cost controls
├── app.py                     # Streamlit UI with Evolution Dashboard tab
└── run_evolution.py           # Main evolution loop
```

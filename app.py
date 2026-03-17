import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.workflow import build_graph  # noqa: E402
from models.schemas import AgentState, TaskStatus  # noqa: E402

st.set_page_config(
    page_title="Multi-Agent Code Generator",
    page_icon="🤖",
    layout="wide",
)

_EXPERIMENTS_DIR = Path(__file__).parent / "experiments"

st.title("Multi-Agent Code Generator")
st.caption("Powered by LangGraph + Claude")

PIPELINE_ORDER = ["orchestrator", "planner", "coder", "reviewer", "tester"]

AGENT_META = {
    "orchestrator": ("🎯", "Orchestrator", "Parsing your request..."),
    "planner": ("📋", "Planner", "Building an implementation plan..."),
    "coder": ("💻", "Coder", "Writing code..."),
    "reviewer": ("🔍", "Reviewer", "Reviewing code quality..."),
    "tester": ("🧪", "Tester", "Running tests..."),
}

CIRCLE = {
    "waiting": "⚪",
    "running": "🟡",
    "done": "🟢",
    "failed": "🔴",
}


TESTER_MODELS = {
    "Fast — Haiku (simple projects)": "claude-haiku-4-5-20251001",
    "Standard — Sonnet (most projects)": "claude-sonnet-4-6",
    "Thorough — Opus (complex projects)": "claude-opus-4-6",
}


def render_sidebar() -> tuple[int, str, dict, dict]:
    """Render sidebar and return (max_iterations, tester_model, placeholders, indicator_states)."""
    placeholders: dict = {}
    indicator_states: dict = {key: "waiting" for key in PIPELINE_ORDER}

    with st.sidebar:
        st.header("Settings")
        max_iterations = st.slider("Max revision iterations", min_value=1, max_value=5, value=3)

        st.markdown("**Tester agent model**")
        tester_choice = st.radio(
            label="tester_model",
            options=list(TESTER_MODELS.keys()),
            index=1,
            label_visibility="collapsed",
            help="Haiku is fastest but may miss edge cases. Opus is most thorough but costlier.",
        )
        tester_model = TESTER_MODELS[tester_choice]

        st.divider()
        st.markdown("**Pipeline Status**")
        for key in PIPELINE_ORDER:
            emoji, label, _ = AGENT_META[key]
            circle_col, label_col = st.columns([1, 5])
            with circle_col:
                placeholders[key] = st.empty()
                placeholders[key].markdown(CIRCLE["waiting"])
            with label_col:
                st.markdown(f"{emoji} {label}")

    return max_iterations, tester_model, placeholders, indicator_states


def set_indicator(placeholders: dict, indicator_states: dict, agent: str, state: str) -> None:
    indicator_states[agent] = state
    placeholders[agent].markdown(CIRCLE[state])


def run_with_streaming(
    request: str,
    max_iterations: int,
    tester_model: str,
    placeholders: dict,
    indicator_states: dict,
) -> AgentState:
    """Run the graph with app.stream() and update sidebar indicators in real time."""
    os.environ["TESTER_MODEL"] = tester_model

    graph = build_graph()
    recursion_limit = int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "50"))

    initial_state = AgentState(
        user_request=request,
        max_iterations=max_iterations,
    )

    # Mark the first agent as running before stream starts
    set_indicator(placeholders, indicator_states, "orchestrator", "running")

    final_state: dict = {}
    coder_iteration = 0

    with st.status("Running agent pipeline...", expanded=True) as pipeline_status:
        try:
            for chunk in graph.stream(
                initial_state,
                config={"recursion_limit": recursion_limit},
            ):
                for node_name, state_update in chunk.items():
                    emoji, label, description = AGENT_META.get(
                        node_name, ("⚙️", node_name, "Running...")
                    )

                    # Mark this agent done
                    set_indicator(placeholders, indicator_states, node_name, "done")

                    # Log to the streaming status panel
                    if node_name == "coder":
                        coder_iteration += 1
                        suffix = f" (revision {coder_iteration})" if coder_iteration > 1 else ""
                        st.write(f"{emoji} **{label}{suffix}** — {description}")
                    else:
                        st.write(f"{emoji} **{label}** — {description}")

                    final_state.update(state_update)

                    # Determine which agent runs next and mark it yellow
                    if node_name == "tester":
                        test = state_update.get("test_result")
                        review = final_state.get("review")
                        iteration_count = final_state.get("iteration", 0)
                        will_revise = (
                            (test is not None and not test.passed)
                            or (review is not None and not review.approved)
                        ) and iteration_count < max_iterations

                        if will_revise:
                            # Reset coder → reviewer → tester back to waiting, then coder to running
                            for agent in ["coder", "reviewer", "tester"]:
                                set_indicator(placeholders, indicator_states, agent, "waiting")
                            set_indicator(placeholders, indicator_states, "coder", "running")
                    else:
                        idx = PIPELINE_ORDER.index(node_name)
                        if idx + 1 < len(PIPELINE_ORDER):
                            next_agent = PIPELINE_ORDER[idx + 1]
                            set_indicator(placeholders, indicator_states, next_agent, "running")

        except Exception:
            # Mark any still-running agent as failed
            for agent, state in indicator_states.items():
                if state == "running":
                    set_indicator(placeholders, indicator_states, agent, "failed")
            pipeline_status.update(label="Pipeline failed.", state="error", expanded=True)
            raise

        pipeline_status.update(label="Pipeline complete!", state="complete", expanded=False)

    return AgentState(**{k: v for k, v in final_state.items() if v is not None})


def render_results(state: AgentState) -> None:
    status_color = {
        TaskStatus.COMPLETED: "green",
        TaskStatus.FAILED: "red",
        TaskStatus.NEEDS_REVISION: "orange",
        TaskStatus.IN_PROGRESS: "blue",
        TaskStatus.PENDING: "gray",
    }
    color = status_color.get(state.status, "gray")
    st.markdown(f"**Status:** :{color}[{state.status.value.upper()}]")

    col1, col2 = st.columns(2)

    with col1:
        if state.plan:
            with st.expander("📋 Plan", expanded=False):
                st.markdown(f"**Objective:** {state.plan.objective}")
                st.markdown("**Steps:**")
                for i, step in enumerate(state.plan.steps, 1):
                    st.markdown(f"{i}. {step}")
                if state.plan.dependencies:
                    st.markdown(f"**Dependencies:** `{'`, `'.join(state.plan.dependencies)}`")

        if state.review:
            with st.expander(f"🔍 Review — {state.review.score}/10", expanded=False):
                approved_icon = "✅" if state.review.approved else "❌"
                st.markdown(f"{approved_icon} **{state.review.summary}**")
                if state.review.issues:
                    st.markdown("**Issues:**")
                    for issue in state.review.issues:
                        st.markdown(f"- {issue}")
                if state.review.suggestions:
                    st.markdown("**Suggestions:**")
                    for s in state.review.suggestions:
                        st.markdown(f"- {s}")

    with col2:
        if state.test_result:
            passed_icon = "✅" if state.test_result.passed else "❌"
            label = (
                f"🧪 Tests — {passed_icon} "
                f"{state.test_result.passed_tests}/{state.test_result.total_tests}"
            )
            with st.expander(label, expanded=False):
                if state.test_result.errors:
                    st.markdown("**Failures:**")
                    for err in state.test_result.errors:
                        st.code(err)
                if state.test_result.output:
                    st.text(state.test_result.output)

    if state.artifacts:
        st.subheader(f"Generated Code ({len(state.artifacts)} file(s))")
        for artifact in state.artifacts:
            with st.expander(f"`{artifact.filename}` — {artifact.description}", expanded=True):
                st.code(artifact.content, language=artifact.language)
                st.download_button(
                    label=f"Download {artifact.filename}",
                    data=artifact.content,
                    file_name=artifact.filename,
                    mime="text/plain",
                    key=artifact.filename,
                )


def _list_experiments() -> list[str]:
    """Return experiment names found in the experiments/ directory."""
    if not _EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [d.name for d in _EXPERIMENTS_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )


def _load_evolution_history(experiment_name: str) -> dict | None:
    """Load evolution_history.json for the given experiment, or None if missing."""
    history_path = _EXPERIMENTS_DIR / experiment_name / "evolution_history.json"
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def render_evolution_tab() -> None:
    """Render the Evolution Dashboard tab content."""
    st.subheader("Evolution Dashboard")
    st.caption("Inspect performance across generations for any saved experiment.")

    experiments = _list_experiments()
    if not experiments:
        st.info(
            "No experiments found. Run the evolution loop first:\n"
            "```\npython run_evolution.py --generations 3 --batch-size 3\n```"
        )
        return

    selected = st.selectbox("Select experiment", experiments)
    if not selected:
        return

    history = _load_evolution_history(selected)
    if history is None:
        st.warning(f"No evolution_history.json found for experiment '{selected}'.")
        return

    generations_data = history.get("generations", [])
    if not generations_data:
        st.warning("No generation data found in this experiment.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown(f"**Experiment:** `{selected}` — {len(generations_data)} generation(s)")

    best = max(generations_data, key=lambda g: g.get("overall_score", 0))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Generations run", len(generations_data))
    col2.metric("Best overall score", f"{best.get('overall_score', 0):.3f}")
    col3.metric("Best bug detection", f"{best.get('bug_detection_rate', 0):.1%}")
    col4.metric("Best gen #", best.get("generation", "—"))

    # ── Performance chart ────────────────────────────────────────────────────
    chart_path = _EXPERIMENTS_DIR / selected / "evolution_chart.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Evolution performance chart", use_column_width=True)
    else:
        st.info("No chart found. The chart is generated at the end of a full evolution run.")

    # ── Per-generation metrics table ─────────────────────────────────────────
    st.markdown("#### Per-Generation Metrics")
    rows = []
    for g in generations_data:
        rows.append(
            {
                "Gen": g.get("generation"),
                "Overall": f"{g.get('overall_score', 0):.3f}",
                "Bug detection": f"{g.get('bug_detection_rate', 0):.1%}",
                "False failure": f"{g.get('false_failure_rate', 0):.1%}",
                "Redundancy": f"{g.get('redundancy_rate', 0):.1%}",
                "Coverage quality": f"{g.get('coverage_quality', 0):.1f}",
                "Edge case": f"{g.get('edge_case_coverage', 0):.1f}",
            }
        )
    st.table(rows)

    # ── Prompt comparison ────────────────────────────────────────────────────
    st.markdown("#### Prompt Viewer")
    prompt_versions: dict = history.get("prompt_versions", {})
    gen_keys = sorted(prompt_versions.keys(), key=lambda k: int(k))

    if gen_keys:
        col_a, col_b = st.columns(2)
        with col_a:
            gen_a = st.selectbox("Generation A", gen_keys, index=0, key="gen_a")
        with col_b:
            default_b = min(1, len(gen_keys) - 1)
            gen_b = st.selectbox("Generation B", gen_keys, index=default_b, key="gen_b")

        col_left, col_right = st.columns(2)
        with col_left:
            st.caption(f"Gen {gen_a}")
            st.text_area(
                label=f"Prompt gen {gen_a}",
                value=prompt_versions.get(str(gen_a), ""),
                height=300,
                key=f"prompt_{gen_a}",
                label_visibility="collapsed",
            )
        with col_right:
            st.caption(f"Gen {gen_b}")
            st.text_area(
                label=f"Prompt gen {gen_b}",
                value=prompt_versions.get(str(gen_b), ""),
                height=300,
                key=f"prompt_{gen_b}",
                label_visibility="collapsed",
            )

    # ── Strengths / weaknesses per generation ────────────────────────────────
    st.markdown("#### Qualitative Observations")
    for g in generations_data:
        gen_num = g.get("generation")
        with st.expander(f"Gen {gen_num} — overall {g.get('overall_score', 0):.3f}"):
            obs_col1, obs_col2 = st.columns(2)
            with obs_col1:
                st.markdown("**Strengths**")
                for s in g.get("strengths", []):
                    st.markdown(f"- {s}")
            with obs_col2:
                st.markdown("**Weaknesses**")
                for w in g.get("weaknesses", []):
                    st.markdown(f"- {w}")


def main() -> None:
    """App entry point — renders Code Generator and Evolution Dashboard tabs."""
    tab_gen, tab_evo = st.tabs(["Code Generator", "Evolution Dashboard"])

    with tab_gen:
        max_iterations, tester_model, placeholders, indicator_states = render_sidebar()

        request = st.text_area(
            "Describe what you want to build",
            placeholder="e.g. A Python FastAPI server with a /health and /echo POST endpoints",
            height=120,
        )

        if st.button("Generate Code", type="primary", disabled=not request.strip()):
            if not os.getenv("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
                return

            try:
                state = run_with_streaming(
                    request.strip(), max_iterations, tester_model, placeholders, indicator_states
                )
                render_results(state)
            except Exception as e:
                st.error(f"Pipeline error: {e}")

    with tab_evo:
        render_evolution_tab()


if __name__ == "__main__":
    main()

"""Streamlit UI -- side-by-side comparison of agent paradigms."""

from __future__ import annotations

import concurrent.futures
import time
import traceback

import pandas as pd
import streamlit as st

# ── bootstrap: must be importable from project root ───────────────
import sys, pathlib  # noqa: E401
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agents import AGENT_REGISTRY, AgentResult, BaseAgent  # noqa: E402
from llm import QwenClient, GLMClient, DeepSeekClient  # noqa: E402
from tools import ToolRegistry, CalculatorTool, WikiSearchTool  # noqa: E402
import config  # noqa: E402

# ── page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Lookbook",
    page_icon="🔬",
    layout="wide",
)

# ── .env defaults per provider ────────────────────────────────────
_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    "Qwen": {"api_key": config.QWEN_API_KEY, "model": config.QWEN_MODEL},
    "GLM": {"api_key": config.GLM_API_KEY, "model": config.GLM_MODEL},
    "DeepSeek": {"api_key": config.DEEPSEEK_API_KEY, "model": config.DEEPSEEK_MODEL},
}

# ── sidebar: LLM provider config ─────────────────────────────────
st.sidebar.title("Agent Lookbook")
st.sidebar.markdown("Compare agent paradigms side by side.")

provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Qwen", "GLM", "DeepSeek"],
    index=0,
)

_defaults = _ENV_DEFAULTS.get(provider, {})
_env_key = _defaults.get("api_key", "")
_env_model = _defaults.get("model", "")

api_key = st.sidebar.text_input(
    "API Key",
    value=_env_key,
    type="password",
    help="Auto-filled from .env if available. Override as needed.",
)

model_override = st.sidebar.text_input(
    "Model",
    value=_env_model,
    help="Auto-filled from .env. Change to use a different model.",
)

if _env_key:
    st.sidebar.caption(f"✅ API key loaded from .env for {provider}")

# ── sidebar: paradigm selector ────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("Paradigms")

available = list(AGENT_REGISTRY.keys())
selected_paradigms = []
for name in available:
    agent_cls = AGENT_REGISTRY[name]
    if st.sidebar.checkbox(name, value=True, help=agent_cls.paradigm_description):
        selected_paradigms.append(name)

if not selected_paradigms:
    st.warning("Please select at least one paradigm from the sidebar.")
    st.stop()


# ── helper: build LLM client ─────────────────────────────────────
def _build_llm():
    kwargs: dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model_override:
        kwargs["model"] = model_override

    match provider:
        case "Qwen":
            return QwenClient(**kwargs)
        case "GLM":
            return GLMClient(**kwargs)
        case "DeepSeek":
            return DeepSeekClient(**kwargs)
        case _:
            raise ValueError(f"Unknown provider: {provider}")


def _build_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(WikiSearchTool())
    return registry


# ── helper: run one agent ─────────────────────────────────────────
def _run_agent(paradigm_name: str, query: str) -> tuple[str, AgentResult | str]:
    """Returns (paradigm_name, AgentResult) or (paradigm_name, error_string)."""
    try:
        llm = _build_llm()
        tools = _build_tools()
        agent_cls = AGENT_REGISTRY[paradigm_name]
        agent: BaseAgent = agent_cls(llm=llm, tools=tools)

        # Time the agent run
        start_time = time.time()
        result = agent.run(query)
        elapsed = time.time() - start_time

        # Extract metrics from LLM client
        stats = llm.get_stats()
        result.elapsed_time = elapsed
        result.llm_calls = stats["call_count"]
        result.token_usage = {
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["completion_tokens"],
            "total_tokens": stats["total_tokens"],
        }

        return paradigm_name, result
    except Exception:
        return paradigm_name, traceback.format_exc()


# ── step badge colors ─────────────────────────────────────────────
_STEP_COLORS = {
    "thought": "🧠",
    "action": "🔧",
    "observation": "👁️",
    "reflection": "🪞",
    "plan": "📋",
    "answer": "✅",
    "response": "💬",
    "error": "❌",
}


def _render_collapsed_card(name: str, result: AgentResult | str) -> None:
    """Render a collapsed card showing only summary."""
    desc = AGENT_REGISTRY[name].paradigm_description
    st.markdown(f"**{name}**")
    st.caption(desc)

    if isinstance(result, str):
        st.error("Failed")
    else:
        # Show answer summary (first 80 chars)
        answer_preview = result.answer[:80] + "..." if len(result.answer) > 80 else result.answer
        st.info(answer_preview)
        # Show key metrics as badges
        st.caption(f"⏱ {result.elapsed_time:.1f}s | 📞 {result.llm_calls} calls | 🎫 {result.token_usage.get('total_tokens', 0)} tokens")


def _render_expanded_card(name: str, result: AgentResult | str) -> None:
    """Render an expanded card showing full thinking process."""
    desc = AGENT_REGISTRY[name].paradigm_description
    st.markdown(f"### {name}")
    st.caption(desc)

    if isinstance(result, str):
        # Error
        st.error("Agent failed")
        st.code(result, language="text")
        return

    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Time", f"{result.elapsed_time:.2f}s")
    col2.metric("LLM Calls", result.llm_calls)
    col3.metric("Total Tokens", result.token_usage.get("total_tokens", 0))

    # Thinking process
    st.markdown("#### Thinking Process")
    for step in result.steps:
        icon = _STEP_COLORS.get(step.type, "📝")
        st.markdown(f"{icon} **{step.type.upper()}**")
        st.markdown(step.content)
        if step.metadata:
            st.caption(f"metadata: {step.metadata}")
        st.markdown("---")

    # Final answer
    st.success(f"**Answer:** {result.answer}")


def _render_metrics_table(results: dict[str, AgentResult | str], selected_paradigms: list[str]) -> None:
    """Render a metrics comparison table at the top."""
    rows = []
    for name in selected_paradigms:
        result = results.get(name)
        if isinstance(result, AgentResult):
            rows.append({
                "Agent": name,
                "Time (s)": f"{result.elapsed_time:.2f}",
                "LLM Calls": result.llm_calls,
                "Prompt Tokens": result.token_usage.get("prompt_tokens", 0),
                "Completion Tokens": result.token_usage.get("completion_tokens", 0),
                "Total Tokens": result.token_usage.get("total_tokens", 0),
            })
        else:
            rows.append({
                "Agent": name,
                "Time (s)": "-",
                "LLM Calls": "-",
                "Prompt Tokens": "-",
                "Completion Tokens": "-",
                "Total Tokens": "-",
            })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── session state for horizontal expand ───────────────────────────
if "expanded_agent" not in st.session_state:
    st.session_state.expanded_agent = None
if "results" not in st.session_state:
    st.session_state.results = {}
if "last_selected_paradigms" not in st.session_state:
    st.session_state.last_selected_paradigms = []


def _toggle_expand(name: str) -> None:
    """Toggle expand/collapse for a card."""
    if st.session_state.expanded_agent == name:
        st.session_state.expanded_agent = None
    else:
        st.session_state.expanded_agent = name


# ── main area ─────────────────────────────────────────────────────
st.title("🔬 Agent Lookbook")
st.caption("Send the same query to multiple agent paradigms and compare their reasoning.")

query = st.text_area(
    "Your question",
    placeholder="e.g. What is 23 * 47 + 12? / Explain how photosynthesis works.",
    height=100,
)

run_button = st.button("Run", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not api_key:
        st.warning("Please enter your API key in the sidebar.")
        st.stop()

    # Reset expanded state on new run
    st.session_state.expanded_agent = None
    st.session_state.last_selected_paradigms = selected_paradigms.copy()

    # Run all selected paradigms in parallel
    with st.spinner("Running agents..."):
        results: dict[str, AgentResult | str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_paradigms)) as pool:
            futures = {
                pool.submit(_run_agent, name, query.strip()): name
                for name in selected_paradigms
            }
            for future in concurrent.futures.as_completed(futures):
                paradigm_name, result = future.result()
                results[paradigm_name] = result

        st.session_state.results = results

# Display results if we have any
if st.session_state.results and st.session_state.last_selected_paradigms:
    results = st.session_state.results
    paradigm_list = st.session_state.last_selected_paradigms
    expanded = st.session_state.expanded_agent

    # ── Metrics comparison table ──────────────────────────────────
    st.markdown("### 📊 Metrics Comparison")
    _render_metrics_table(results, paradigm_list)

    st.markdown("---")
    st.markdown("### 🔍 Agent Results")
    st.caption("Click a card to expand/collapse. Only one card can be expanded at a time.")

    # ── Calculate column widths based on expanded state ───────────
    n = len(paradigm_list)
    if expanded and expanded in paradigm_list:
        # Expanded card gets 3x width, others get 1x
        widths = []
        for name in paradigm_list:
            if name == expanded:
                widths.append(3)
            else:
                widths.append(1)
    else:
        # All equal width
        widths = [1] * n

    # ── Render cards in columns ───────────────────────────────────
    cols = st.columns(widths)
    for col, name in zip(cols, paradigm_list):
        with col:
            result = results.get(name, "No result")
            is_expanded = (expanded == name)

            # Card container with border
            with st.container(border=True):
                # Toggle button
                btn_label = "▼ Collapse" if is_expanded else "▶ Expand"
                if st.button(btn_label, key=f"toggle_{name}", use_container_width=True):
                    _toggle_expand(name)
                    st.rerun()

                if is_expanded:
                    _render_expanded_card(name, result)
                else:
                    _render_collapsed_card(name, result)

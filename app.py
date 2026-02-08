"""Streamlit UI -- side-by-side comparison of agent paradigms."""

from __future__ import annotations

import concurrent.futures
import traceback

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
        result = agent.run(query)
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


def _render_result(name: str, result: AgentResult | str) -> None:
    """Render a single agent's result inside a column."""
    desc = AGENT_REGISTRY[name].paradigm_description
    st.markdown(f"**{name}** — _{desc}_")

    if isinstance(result, str):
        # Error
        st.error("Agent failed")
        st.code(result, language="text")
        return

    # Thinking process
    with st.expander("Thinking Process", expanded=True):
        for step in result.steps:
            icon = _STEP_COLORS.get(step.type, "📝")
            st.markdown(f"{icon} **{step.type.upper()}**")
            st.markdown(step.content)
            if step.metadata:
                st.caption(f"metadata: {step.metadata}")
            st.markdown("---")

    # Final answer
    st.success(result.answer)


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

    # Render results in columns (preserve selection order)
    cols = st.columns(len(selected_paradigms))
    for col, name in zip(cols, selected_paradigms):
        with col:
            _render_result(name, results.get(name, "No result"))

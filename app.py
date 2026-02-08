"""Streamlit UI -- side-by-side comparison of agent paradigms."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
import traceback

import pandas as pd
import streamlit as st

# ── Configure logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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

# ── global custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="stExpander"] {
    border-radius: 8px;
}
/* Clean up default container borders so our accent borders take over */
div[data-testid="stVerticalBlockBorderWrapper"] {
    transition: box-shadow 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

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
    logger.info(f"[{paradigm_name}] Starting...")
    try:
        llm = _build_llm()
        tools = _build_tools()
        agent_cls = AGENT_REGISTRY[paradigm_name]
        agent: BaseAgent = agent_cls(llm=llm, tools=tools)

        # Time the agent run
        start_time = time.time()
        logger.info(f"[{paradigm_name}] Calling agent.run()...")
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

        logger.info(f"[{paradigm_name}] Completed in {elapsed:.2f}s, {stats['call_count']} LLM calls, {stats['total_tokens']} tokens")
        return paradigm_name, result
    except Exception as e:
        logger.error(f"[{paradigm_name}] Failed: {e}")
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

# ── accent color palette per paradigm (muted, modern) ────────────
_ACCENT_COLORS: dict[str, str] = {
    "Vanilla":     "#868e96",   # cool gray
    "CoT":         "#5c7cfa",   # soft indigo
    "ReAct":       "#339af0",   # calm blue
    "Reflexion":   "#845ef7",   # gentle violet
    "ToT":         "#51cf66",   # light green
    "GoT":         "#f06595",   # soft rose
    "CodeAct":     "#22b8cf",   # muted cyan
    "InterCode":   "#ff922b",   # warm tangerine
    "ADaPT":       "#ff6b6b",   # soft coral
    "AdaPlanner":  "#20c997",   # fresh mint
}

def _accent(name: str) -> str:
    """Return the accent color for a paradigm, with a fallback."""
    return _ACCENT_COLORS.get(name, "#5c7cfa")


def _render_collapsed_strip(name: str, result: AgentResult | str) -> None:
    """Render a collapsed book-spine style strip – each letter rotated 90° CW, read top→bottom."""
    color = _accent(name)
    # Build one <span> per character, each rotated 90 degrees clockwise
    letters_html = "".join(
        f'<span style="display:block; transform:rotate(90deg); line-height:1.15;">{ch}</span>'
        if ch != " "
        else '<span style="display:block; height:6px;"></span>'
        for ch in name
    )
    st.markdown(f'''
    <div style="
        min-height: 240px;
        padding: 20px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: {color}0a;
        border-left: 3px solid {color};
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        cursor: pointer;
        transition: box-shadow 0.2s ease;
    ">
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1px;
            font-weight: 700;
            font-size: 13px;
            color: {color};
            letter-spacing: 0.5px;
        ">
            {letters_html}
        </div>
    </div>
    ''', unsafe_allow_html=True)


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

    # Metrics at the bottom as small text
    st.caption(f"⏱ {result.elapsed_time:.2f}s | 📞 {result.llm_calls} calls | 🎫 {result.token_usage.get('total_tokens', 0)} tokens")


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
# Use collapsed_agents set - default is empty (all expanded)
if "collapsed_agents" not in st.session_state:
    st.session_state.collapsed_agents = set()
if "results" not in st.session_state:
    st.session_state.results = {}
if "last_selected_paradigms" not in st.session_state:
    st.session_state.last_selected_paradigms = []
if "running" not in st.session_state:
    st.session_state.running = False
if "run_query" not in st.session_state:
    st.session_state.run_query = ""
if "run_paradigms" not in st.session_state:
    st.session_state.run_paradigms = []


def _toggle_collapse(name: str) -> None:
    """Toggle collapse/expand for a card."""
    if name in st.session_state.collapsed_agents:
        st.session_state.collapsed_agents.discard(name)
    else:
        st.session_state.collapsed_agents.add(name)


# ── load GSM8K training questions ─────────────────────────────────
_GSM8K_PATH = pathlib.Path(__file__).resolve().parent / "grade-school-math" / "grade_school_math" / "data" / "train.jsonl"


@st.cache_data
def _load_gsm8k_questions() -> list[str]:
    """Load questions from the GSM8K train.jsonl file."""
    questions: list[str] = []
    if _GSM8K_PATH.exists():
        with open(_GSM8K_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        questions.append(obj["question"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return questions


_gsm8k_questions = _load_gsm8k_questions()

# ── main area ─────────────────────────────────────────────────────
st.title("🔬 Agent Paradigm Lookbook")
st.caption("Send the same query to multiple agent paradigms and compare their reasoning.")

# Build dropdown options: "Custom" + GSM8K questions (truncated for display)
_CUSTOM_OPTION = "✏️ Custom (type your own question)"
_display_options = [_CUSTOM_OPTION] + [
    f"Q{i+1}: {q[:90]}{'…' if len(q) > 90 else ''}"
    for i, q in enumerate(_gsm8k_questions)
]

_is_running = st.session_state.running

selected_option = st.selectbox(
    "Select a question",
    options=_display_options,
    index=0,
    disabled=_is_running,
    help="Choose a question from the GSM8K training set, or select 'Custom' to type your own.",
)

if selected_option == _CUSTOM_OPTION:
    query = st.text_area(
        "Your question",
        placeholder="e.g. What is 23 * 47 + 12? / Explain how photosynthesis works.",
        height=100,
        disabled=_is_running,
    )
else:
    # Extract the index from the selected option (format: "Q{i+1}: ...")
    _selected_idx = _display_options.index(selected_option) - 1  # -1 for the Custom option
    query = _gsm8k_questions[_selected_idx]
    st.info(f"**Selected question:** {query}")

run_button = st.button("Run", type="primary", use_container_width=True, disabled=_is_running)

# ── Phase 1: user clicks Run → save query, set running, rerun ────
if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
        st.stop()

    if not api_key:
        st.warning("Please enter your API key in the sidebar.")
        st.stop()

    # Save the query and paradigms, flip running flag
    st.session_state.running = True
    st.session_state.run_query = query.strip()
    st.session_state.run_paradigms = selected_paradigms.copy()
    st.rerun()  # rerun so widgets render as disabled

# ── Phase 2: running flag is set → execute agents ────────────────
if _is_running:
    _run_q = st.session_state.run_query
    _run_p = st.session_state.run_paradigms

    # Reset collapsed state on new run (all expanded by default)
    st.session_state.collapsed_agents = set()
    st.session_state.last_selected_paradigms = _run_p

    with st.spinner("Running agents..."):
        results: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(_run_p)) as pool:
            futures = {
                pool.submit(_run_agent, name, _run_q): name
                for name in _run_p
            }
            for future in concurrent.futures.as_completed(futures):
                paradigm_name, result = future.result()
                results[paradigm_name] = result

        st.session_state.results = results

    # Done → unlock widgets and rerun
    st.session_state.running = False
    st.rerun()

# Display results if we have any
if st.session_state.results and st.session_state.last_selected_paradigms:
    results = st.session_state.results
    paradigm_list = st.session_state.last_selected_paradigms
    collapsed = st.session_state.collapsed_agents

    # ── Agent Results (cards first) ───────────────────────────────
    st.markdown("### 🔍 Agent Results")
    st.caption("Click a card to collapse/expand.")

    # ── Calculate column widths based on collapsed state ──────────
    # Collapsed cards get width 1 (narrow strip), expanded get width 6
    widths = []
    for name in paradigm_list:
        if name in collapsed:
            widths.append(1)  # narrow book-spine
        else:
            widths.append(6)  # full width

    # ── Inject accent CSS for expanded cards (marker + :has selector) ─
    card_css_rules = ""
    for name in paradigm_list:
        c = _accent(name)
        safe = name.replace(" ", "-").lower()
        card_css_rules += f"""
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.card-marker-{safe}) {{
            border: 1px solid rgba(0,0,0,0.08) !important;
            border-top: 3px solid {c} !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
        }}
        """
    st.markdown(f"<style>{card_css_rules}</style>", unsafe_allow_html=True)

    # ── Render cards in columns ───────────────────────────────────
    cols = st.columns(widths)
    for col, name in zip(cols, paradigm_list):
        with col:
            result = results.get(name, "No result")
            is_collapsed = (name in collapsed)
            color = _accent(name)
            safe = name.replace(" ", "-").lower()

            if is_collapsed:
                # Collapsed spine strip – entire thing is clickable
                _render_collapsed_strip(name, result)
                if st.button(name, key=f"toggle_{name}", use_container_width=True):
                    _toggle_collapse(name)
                    st.rerun()
            else:
                # Expanded card with glow border
                with st.container(border=True):
                    # Invisible marker so CSS :has() can target this container
                    st.markdown(
                        f'<div class="card-marker-{safe}" style="display:none;"></div>',
                        unsafe_allow_html=True,
                    )
                    # Clickable header to collapse
                    if st.button(f"▼ {name}", key=f"toggle_{name}", use_container_width=True, type="tertiary"):
                        _toggle_collapse(name)
                        st.rerun()
                    _render_expanded_card(name, result)

    # ── Metrics comparison table (below cards) ────────────────────
    st.markdown("---")
    st.markdown("### 📊 Metrics Comparison")
    _render_metrics_table(results, paradigm_list)

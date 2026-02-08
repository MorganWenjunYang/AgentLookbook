"""ReCode agent -- Recursive Code Generation for unified planning & action.

Core idea (FoundationAgents, 2025 -- ICLR submission):
    Unifies planning and action through **recursive code generation**.
    High-level plans are abstract placeholder functions that are
    recursively decomposed into finer-grained sub-functions until
    reaching primitive, executable actions.

    The algorithm maintains a **code tree** where each node is either:
        - A **stub** (placeholder function needing expansion)
        - **Executable** code (primitive actions)

    Loop:
        1. Pick the next stub node in the tree.
        2. **Expand**: ask LLM to implement it as either:
           a. Direct code (1-2 primitive operations) -> execute
           b. Smaller placeholder sub-functions -> recurse
        3. **Execute** the generated code.
        4. If execution calls a placeholder, mark it as stub and repeat.
        5. Continue until all nodes are completed or max depth is reached.

    This dissolves the rigid plan/act boundary and achieves effectively
    infinite action spaces without predefined sets.

    Ref paper:  https://arxiv.org/abs/2510.23564
    Ref impl:   https://github.com/FoundationAgents/ReCode

Implementation:
    Adapted for general QA (no interactive environment):
    - Root task: ``solve(question)`` as initial stub.
    - Expand: LLM either writes direct Python code to compute the answer
      or breaks it into sub-function stubs.
    - Execute: code runs in an in-process Python REPL (persistent state).
    - Tools from ToolRegistry are injected as callable functions.
    - A ``submit(answer)`` function captures the final answer.
    - All expansion/execution steps logged as ThinkStep.
"""

from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from enum import Enum

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXPAND_SYSTEM = """\
You are the EXPAND step in a recursive code-generation agent.
Your job is to replace a placeholder function with its implementation.

Available tools (call them as Python functions):
{tool_signatures}

Decision rules:
- If the task can be solved in 1-3 lines of Python, write the code directly.
  Use ``print()`` to output results or call ``submit(answer)`` to give the
  final answer.
- If the task requires more steps, break it into smaller placeholder function
  calls.  Placeholder calls look like::

      result = descriptive_function_name(arg1, arg2)

  These will be expanded in later turns.

Format your response EXACTLY as:

<think>
Brief reasoning about how to implement or decompose this function.
</think>

<execute>
# Python code here -- either direct implementation or placeholder calls
</execute>

Rules:
- Do NOT use ``def`` or ``async def`` -- only write top-level statements.
- Placeholder function names should be descriptive snake_case.
- Call ``submit(your_answer)`` when you have the final answer.
"""

EXPAND_USER = """\
Task to expand:
{task_code}

Context from previous steps:
{context}

Expand this task:"""


# ---------------------------------------------------------------------------
# Code tree node
# ---------------------------------------------------------------------------

class _NodeStatus(Enum):
    STUB = "stub"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class _CodeNode:
    """One node in the recursive code tree."""
    code: str
    depth: int = 0
    status: _NodeStatus = _NodeStatus.STUB
    thought: str = ""
    output: str = ""
    error: str = ""
    children: list[_CodeNode] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"CodeNode(depth={self.depth}, status={self.status.value}, code={self.code[:50]!r})"


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

class _PythonREPL:
    """Minimal in-process Python REPL with persistent namespace."""

    def __init__(self, namespace: dict | None = None) -> None:
        self.namespace: dict = namespace or {}
        self.called_stubs: list[str] = []  # placeholder calls detected

    def run(self, code: str) -> dict:
        """Execute code, return {success, stdout, error, stubs}."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        self.called_stubs = []

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)  # noqa: S102
        except NameError as e:
            # A NameError likely means a placeholder function was called
            name_match = re.search(r"name '(\w+)' is not defined", str(e))
            if name_match:
                stub_name = name_match.group(1)
                self.called_stubs.append(stub_name)
                return {
                    "success": False,
                    "stdout": stdout_buf.getvalue(),
                    "error": f"NeedExpansion: {stub_name}",
                    "stubs": [stub_name],
                }
            return {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "error": traceback.format_exc(limit=4),
                "stubs": [],
            }
        except Exception:
            return {
                "success": False,
                "stdout": stdout_buf.getvalue(),
                "error": traceback.format_exc(limit=4),
                "stubs": [],
            }

        return {
            "success": True,
            "stdout": stdout_buf.getvalue(),
            "error": "",
            "stubs": [],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_xml_tag(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>."""
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _extract_placeholder_calls(code: str) -> list[str]:
    """Find placeholder function calls (snake_case names that aren't builtins)."""
    builtins = {"print", "submit", "len", "str", "int", "float", "list",
                "dict", "range", "enumerate", "zip", "sorted", "max", "min",
                "abs", "sum", "round", "type", "isinstance", "bool", "set",
                "tuple", "map", "filter", "open", "input", "format", "repr"}
    # Match function calls like: result = some_function(...)  or  some_function(...)
    pattern = re.compile(r"(?:\w+\s*=\s*)?(\w+)\s*\(")
    found = set()
    for m in pattern.finditer(code):
        name = m.group(1)
        if name not in builtins and "_" in name:  # placeholder = snake_case
            found.add(name)
    return list(found)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent
class ReCodeAgent(BaseAgent):
    paradigm_name = "ReCode"
    paradigm_description = "Recursive code generation: decompose tasks into a code tree, expand stubs, execute."

    max_depth: int = 4
    max_expand_retries: int = 3

    def run(self, query: str) -> AgentResult:
        steps: list[ThinkStep] = []

        # ── Set up REPL with tools + submit ──────────────────────
        submitted: list[str] = []

        def submit(answer):
            submitted.append(str(answer))
            print(f"SUBMITTED: {answer}")

        namespace: dict = {"submit": submit, "__builtins__": __builtins__}
        # Inject tools
        if self.tools:
            for tool in self.tools.list_tools():
                namespace[tool.name] = lambda inp, _t=tool: _t.run(str(inp))

        repl = _PythonREPL(namespace)

        # ── Build tool signatures for prompt ─────────────────────
        tool_sigs = self.tools.describe_all() if self.tools else "No tools available."

        # ── Create root node ─────────────────────────────────────
        root = _CodeNode(code=f"solve('{query}')", depth=0)
        steps.append(ThinkStep(
            type="plan",
            content=f"[Root] {root.code}",
            metadata={"depth": 0},
        ))

        # ── Process stub queue (BFS) ─────────────────────────────
        stub_queue: list[_CodeNode] = [root]
        context_lines: list[str] = []

        while stub_queue:
            node = stub_queue.pop(0)

            if node.depth >= self.max_depth:
                steps.append(ThinkStep(
                    type="reflection",
                    content=f"[Depth {node.depth}] Max depth reached for: {node.code[:80]}",
                ))
                # Force a direct answer attempt
                node.code = f"submit('Unable to solve within depth limit: {query}')"

            # ── Expand stub ───────────────────────────────────────
            steps.append(ThinkStep(
                type="action",
                content=f"[Expand depth={node.depth}] {node.code[:120]}",
                metadata={"depth": node.depth},
            ))

            expanded_code = None
            for attempt in range(1, self.max_expand_retries + 1):
                expand_msgs = [
                    {"role": "system", "content": EXPAND_SYSTEM.format(tool_signatures=tool_sigs)},
                    {"role": "user", "content": EXPAND_USER.format(
                        task_code=node.code,
                        context="\n".join(context_lines[-10:]) if context_lines else "(start of task)",
                    )},
                ]
                response = self.llm.chat(expand_msgs, temperature=0.2)

                thought = _parse_xml_tag(response, "think")
                code = _parse_xml_tag(response, "execute")

                if not code:
                    # Fallback: try to extract code block
                    code_m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
                    if code_m:
                        code = code_m.group(1).strip()
                    else:
                        code = response.strip()

                if thought:
                    node.thought = thought
                    steps.append(ThinkStep(
                        type="thought",
                        content=f"[Think] {thought[:300]}",
                        metadata={"depth": node.depth},
                    ))

                # Validate: no def statements allowed
                if re.search(r"^\s*(async\s+)?def\s+", code, re.MULTILINE):
                    steps.append(ThinkStep(
                        type="reflection",
                        content=f"[Expand retry {attempt}] Code contains def -- not allowed, retrying.",
                    ))
                    continue

                expanded_code = code
                break

            if not expanded_code:
                node.status = _NodeStatus.ERROR
                node.error = "Failed to expand after retries"
                steps.append(ThinkStep(type="error", content=f"[Error] Failed to expand: {node.code[:80]}"))
                continue

            # ── Check for placeholder sub-functions ────────────────
            placeholders = _extract_placeholder_calls(expanded_code)
            # Remove known functions from placeholders
            known = set(namespace.keys())
            real_placeholders = [p for p in placeholders if p not in known]

            if real_placeholders:
                # This is a decomposition step -- create child stubs
                steps.append(ThinkStep(
                    type="plan",
                    content=f"[Decompose] Found {len(real_placeholders)} sub-tasks: {real_placeholders}",
                    metadata={"depth": node.depth, "sub_tasks": real_placeholders},
                ))

                # Define placeholder functions that capture results
                for ph in real_placeholders:
                    # Create a stub function that will be replaced later
                    child = _CodeNode(code=f"{ph}(...)", depth=node.depth + 1)
                    node.children.append(child)
                    stub_queue.append(child)

                    # Define a temporary function that records the call
                    def _make_placeholder(name):
                        def _ph(*args, **kwargs):
                            raise NameError(f"name '{name}' is not defined")
                        return _ph
                    namespace[ph] = _make_placeholder(ph)

                # Try to execute -- it will hit the placeholder and fail
                # We'll execute after child stubs are resolved
                node.code = expanded_code
                node.status = _NodeStatus.STUB
                # Re-queue this node to execute after children
                # Actually, let's execute children first, then come back
                # Put parent at end of queue
                stub_queue.append(node)
                continue

            # ── Execute directly ──────────────────────────────────
            node.code = expanded_code
            node.status = _NodeStatus.RUNNING

            result = repl.run(expanded_code)
            node.output = result["stdout"]

            if result["success"]:
                node.status = _NodeStatus.COMPLETED
                output_text = result["stdout"].strip()
                if output_text:
                    context_lines.append(f"[Step result] {output_text[:200]}")
                steps.append(ThinkStep(
                    type="observation",
                    content=f"[Execute OK] {output_text[:300]}" if output_text else "[Execute OK] (no output)",
                    metadata={"depth": node.depth},
                ))
            elif "NeedExpansion" in result.get("error", ""):
                # A placeholder was called -- create stub
                stub_name = result["stubs"][0] if result["stubs"] else "unknown"
                child = _CodeNode(code=f"{stub_name}(...)", depth=node.depth + 1)
                node.children.append(child)
                stub_queue.append(child)
                # Re-queue parent for re-execution after child resolves
                stub_queue.append(node)
                steps.append(ThinkStep(
                    type="plan",
                    content=f"[NeedExpansion] Stub '{stub_name}' detected, queuing for expansion.",
                    metadata={"depth": node.depth + 1},
                ))
            else:
                node.status = _NodeStatus.ERROR
                node.error = result["error"]
                steps.append(ThinkStep(
                    type="observation",
                    content=f"[Execute Error]\n{result['error'][:300]}",
                    metadata={"depth": node.depth},
                ))
                context_lines.append(f"[Error] {result['error'][:100]}")

            # ── Check if answer was submitted ─────────────────────
            if submitted:
                answer = submitted[-1]
                steps.append(ThinkStep(type="answer", content=answer))
                return AgentResult(answer=answer, steps=steps)

        # ── Fallback if no submit() was called ────────────────────
        # Use the last context as answer
        fallback = context_lines[-1] if context_lines else "Unable to produce an answer."
        steps.append(ThinkStep(type="answer", content=fallback))
        return AgentResult(answer=fallback, steps=steps)

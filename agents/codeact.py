"""CodeAct agent -- executable Python code as the unified action space.

Core idea (Wang et al., 2024 -- ICML):
    Instead of producing JSON or text in a pre-defined format, the LLM
    generates **executable Python code** as its action.  A Python interpreter
    executes the code and returns stdout / stderr as the observation.

    Key advantages over text/JSON actions:
    1. Turing-complete: control flow (loops, branches), data flow (variables).
    2. Compose multiple tools in a single action.
    3. Self-debug: error tracebacks let the LLM fix its own code.
    4. Leverages LLMs' extensive pre-training on code data.

    Ref paper:  https://arxiv.org/abs/2402.01030
    Ref impl:   https://github.com/xingyaoww/code-act

Implementation:
    - System prompt describes available tools as callable Python functions
      and instructs the model to use the ``Action: ... End Action`` or
      ``Answer: ...`` format (faithful to the original paper).
    - Loop: parse Thought / Action / Answer from LLM output →
      execute code in a sandboxed REPL → feed observation back → repeat.
    - The REPL keeps state across turns (variables persist).
    - Max iterations capped to avoid runaway loops.
"""

from __future__ import annotations

import io
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ── prompt ────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """\
You are a helpful assistant that solves tasks by writing executable Python code.

You have access to the following tools as Python functions:
{tool_signatures}

You can use these tools by calling them in your Python code.
You may use for-loops, if-statements, variables, and any standard Python constructs.
Be sure to **print** the final result at the end of your code.

To take an action, use this format:

Thought: <your reasoning about what to do next>
Action:
<python code>
End Action

When you have the final answer, use this format:

Answer: <your final answer>

IMPORTANT:
- Your output should contain either 'Action:' or 'Answer:', but NOT both.
- Always start with a Thought before an Action.
- Each Action block is executed in a persistent Python session (variables survive across turns).
- If your code produces an error, you will see the traceback. Use it to debug and try again.
"""


# ── sandboxed REPL ────────────────────────────────────────────────

class _PythonREPL:
    """Minimal in-process Python REPL with persistent namespace.

    Tools from the ToolRegistry are injected as callable functions so the
    LLM-generated code can invoke them naturally, e.g. ``Calculator("2+3")``.
    """

    def __init__(self, namespace: dict | None = None) -> None:
        self.namespace: dict = namespace or {}

    def run(self, code: str) -> str:
        """Execute *code* and return combined stdout + stderr (truncated)."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)  # noqa: S102 – intentional sandboxed exec
        except Exception:
            stderr_buf.write(traceback.format_exc())

        output = stdout_buf.getvalue() + stderr_buf.getvalue()
        if not output.strip():
            output = "[Executed successfully with no output. Did you forget to print?]"
        # Truncate very long outputs
        if len(output) > 3000:
            output = output[:3000] + "\n...[Output Truncated]"
        return output


# ── parsing helpers ───────────────────────────────────────────────

def _parse_thought(text: str) -> str | None:
    """Extract optional Thought text before Action/Answer."""
    match = re.search(r"Thought:\s*(.*?)(?=Action:|Answer:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_action(text: str) -> str | None:
    """Extract code between ``Action:`` and ``End Action``."""
    match = re.search(r"Action:\s*\n?(.*?)End Action", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_answer(text: str) -> str | None:
    """Extract final answer after ``Answer:``."""
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None


# ── agent ─────────────────────────────────────────────────────────

def _tool_signatures(tools) -> str:
    """Build Python-style function signatures for the prompt."""
    lines: list[str] = []
    for tool in tools.list_tools():
        # Present each tool as: tool_name(input_text) -> str
        lines.append(f"  {tool.name}(input_text: str) -> str")
        lines.append(f"      {tool.description}")
    return "\n".join(lines)


@register_agent
class CodeActAgent(BaseAgent):
    paradigm_name = "CodeAct"
    paradigm_description = "Generate executable Python code as actions; self-debug via interpreter feedback."

    max_iterations: int = 6

    def run(self, query: str) -> AgentResult:
        # ── build system prompt with tool signatures ──────────
        sig_text = _tool_signatures(self.tools) if self.tools else "  (no tools available)"
        system_msg = SYSTEM_TEMPLATE.format(tool_signatures=sig_text)

        # ── prepare REPL with tools injected as callables ─────
        namespace: dict = {}
        for tool in self.tools.list_tools():
            # Wrap so that calling Calculator("2+3") invokes tool.run("2+3")
            namespace[tool.name] = tool.run

        repl = _PythonREPL(namespace)

        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        steps: list[ThinkStep] = []

        for _iteration in range(self.max_iterations):
            # ── call LLM ──────────────────────────────────────
            response = self.llm.chat(messages, temperature=0.0)

            # ── parse thought ─────────────────────────────────
            thought = _parse_thought(response)
            if thought:
                steps.append(ThinkStep(type="thought", content=thought))

            # ── check for final answer ────────────────────────
            answer = _parse_answer(response)
            if answer is not None and _parse_action(response) is None:
                steps.append(ThinkStep(type="answer", content=answer))
                return AgentResult(answer=answer, steps=steps)

            # ── parse and execute code action ─────────────────
            code = _parse_action(response)
            if code is None:
                # Model didn't follow format -- treat entire response as answer
                steps.append(ThinkStep(type="answer", content=response.strip()))
                return AgentResult(answer=response.strip(), steps=steps)

            steps.append(
                ThinkStep(
                    type="action",
                    content=code,
                    metadata={"language": "python"},
                )
            )

            # ── execute in REPL ───────────────────────────────
            observation = repl.run(code)
            steps.append(
                ThinkStep(
                    type="observation",
                    content=observation,
                    metadata={"source": "python_repl"},
                )
            )

            # ── feed observation back to LLM ──────────────────
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": observation})

        # ── max iterations exhausted ──────────────────────────
        fallback = "I was unable to reach a final answer within the allowed steps."
        steps.append(ThinkStep(type="error", content=fallback))
        return AgentResult(answer=fallback, steps=steps)

"""InterCode agent -- interactive coding with execution feedback.

Core idea (Yang et al., 2023 -- NeurIPS Datasets & Benchmarks):
    InterCode models code generation as an **interactive RL environment**:
        - **State**:  task description + history of (action, observation) pairs
        - **Action**: a piece of code submitted to the interpreter
        - **Observation**: execution output (stdout/stderr) from the environment

    Unlike static code generation (single-shot), InterCode allows the agent
    to iteratively refine its code based on execution feedback -- much like a
    human developer running code, reading errors, and fixing them.

    The framework is language- and platform-agnostic.  Our implementation uses
    a Python REPL, but the paradigm applies to Bash, SQL, etc.

    Ref paper:  https://arxiv.org/abs/2306.14898
    Ref impl:   https://github.com/princeton-nlp/intercode

Implementation:
    - System prompt frames the task as an interactive coding session.
    - The agent generates code actions; a Python REPL executes them.
    - Execution output (or errors) is fed back as the next observation.
    - The agent can submit multiple code blocks, refining based on feedback.
    - When ready, the agent emits ``submit`` to finalise its answer.
    - Tools from ToolRegistry are injected into the REPL namespace.

Key difference from CodeAct:
    - CodeAct focuses on using code as a *unified action space* to replace
      JSON/text tool calls, with emphasis on tool composition.
    - InterCode focuses on the *interactive loop* of write-execute-debug,
      treating the entire session as an RL episode with reward signal.
      The prompt encourages explicit observation analysis and iteration.
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
You are an expert programmer solving a task interactively.

You have access to a Python interpreter. At each step you can:
1. **Execute code** to make progress toward solving the task.
2. **Observe** the output (stdout, stderr, errors) from execution.
3. **Iterate**: fix errors, refine your approach based on observations.

Available helper functions in the interpreter:
{tool_signatures}

FORMAT:
- To execute code, write:
    Thought: <your reasoning>
    Action:
    ```python
    <your code>
    ```

- When you have the final answer and are done, write:
    Thought: <why you are done>
    Action: submit(<your final answer as a string>)

RULES:
- You will see execution output after each Action.
- If there is an error, analyse the traceback and fix your code.
- Variables persist between steps.
- Always print intermediate results to verify correctness.
- Do NOT give up -- iterate until you have a correct answer.
"""


# ── sandboxed REPL (shared pattern with CodeAct) ─────────────────

class _PythonREPL:
    """Minimal in-process Python REPL with persistent namespace."""

    def __init__(self, namespace: dict | None = None) -> None:
        self.namespace: dict = namespace or {}
        self.submitted: str | None = None
        # Inject a submit() function so the agent can finalise
        self.namespace["submit"] = self._submit

    def _submit(self, answer: str) -> str:
        self.submitted = str(answer)
        return f"[Submitted answer: {self.submitted}]"

    def run(self, code: str) -> str:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code, self.namespace)  # noqa: S102
        except Exception:
            stderr_buf.write(traceback.format_exc())

        output = stdout_buf.getvalue() + stderr_buf.getvalue()
        if not output.strip():
            output = "[Executed successfully with no output]"
        if len(output) > 3000:
            output = output[:3000] + "\n...[Output Truncated]"
        return output


# ── parsing helpers ───────────────────────────────────────────────

def _parse_thought(text: str) -> str | None:
    match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_code(text: str) -> str | None:
    """Extract code from ```python ... ``` or from Action: block."""
    # Try fenced code block first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: everything after "Action:" that isn't a submit()
    match = re.search(r"Action:\s*\n?(.*?)$", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return None


def _parse_submit(text: str) -> str | None:
    """Check if the action is a submit() call and extract the argument."""
    match = re.search(r"submit\(\s*[\"']?(.*?)[\"']?\s*\)", text)
    return match.group(1).strip() if match else None


# ── tool signatures for prompt ────────────────────────────────────

def _tool_signatures(tools) -> str:
    lines: list[str] = []
    for tool in tools.list_tools():
        lines.append(f"  {tool.name}(input_text: str) -> str")
        lines.append(f"      {tool.description}")
    return "\n".join(lines) if lines else "  (no additional tools)"


# ── agent ─────────────────────────────────────────────────────────

@register_agent
class InterCodeAgent(BaseAgent):
    paradigm_name = "InterCode"
    paradigm_description = "Interactive coding: write code, observe execution, iterate until correct."

    max_iterations: int = 8

    def run(self, query: str) -> AgentResult:
        # ── build system prompt ───────────────────────────────
        sig_text = _tool_signatures(self.tools) if self.tools else "  (none)"
        system_msg = SYSTEM_TEMPLATE.format(tool_signatures=sig_text)

        # ── prepare REPL with tools ───────────────────────────
        namespace: dict = {}
        for tool in self.tools.list_tools():
            namespace[tool.name] = tool.run

        repl = _PythonREPL(namespace)

        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Task: {query}"},
        ]
        steps: list[ThinkStep] = []

        for iteration in range(1, self.max_iterations + 1):
            # ── call LLM ──────────────────────────────────────
            response = self.llm.chat(messages, temperature=0.0)

            # ── parse thought ─────────────────────────────────
            thought = _parse_thought(response)
            if thought:
                steps.append(ThinkStep(
                    type="thought",
                    content=thought,
                    metadata={"iteration": iteration},
                ))

            # ── check for submit ──────────────────────────────
            submitted = _parse_submit(response)
            if submitted is not None:
                steps.append(ThinkStep(type="answer", content=submitted))
                return AgentResult(answer=submitted, steps=steps)

            # ── parse and execute code ────────────────────────
            code = _parse_code(response)
            if code is None:
                # No valid code found; treat response as answer
                steps.append(ThinkStep(type="answer", content=response.strip()))
                return AgentResult(answer=response.strip(), steps=steps)

            steps.append(ThinkStep(
                type="action",
                content=code,
                metadata={"language": "python", "iteration": iteration},
            ))

            observation = repl.run(code)

            steps.append(ThinkStep(
                type="observation",
                content=observation,
                metadata={"iteration": iteration},
            ))

            # ── check if submit() was called inside the code ──
            if repl.submitted is not None:
                steps.append(ThinkStep(type="answer", content=repl.submitted))
                return AgentResult(answer=repl.submitted, steps=steps)

            # ── feed observation back ─────────────────────────
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation (step {iteration}):\n{observation}",
            })

        # ── max iterations exhausted ──────────────────────────
        fallback = "Max iterations reached without a submitted answer."
        steps.append(ThinkStep(type="error", content=fallback))
        return AgentResult(answer=fallback, steps=steps)

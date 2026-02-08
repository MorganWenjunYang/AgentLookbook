"""AdaPlanner agent -- adaptive planning from feedback.

Core idea (Sun et al., 2023 -- NeurIPS):
    An LLM generates a structured, step-by-step plan to solve a task.
    Each step includes an expected outcome (assertion).  The agent then
    executes each step, and if the result deviates from expectations, it
    adaptively refines the plan:

        1. **Plan**       -- generate a numbered step-by-step plan with
                             expected outcomes per step.
        2. **Execute**    -- carry out each step (tool call or reasoning).
        3. **In-plan refinement**  -- if a step fails its assertion, ask
                             the LLM to fix *that specific step* in-place.
        4. **Out-of-plan refinement** -- if in-plan refinement doesn't help,
                             ask the LLM to revise the *remaining* plan.

    The plan uses a structured (code-style) format inspired by the original
    paper, which reduces hallucination.

    Ref paper:  https://arxiv.org/abs/2310.08893
    Ref impl:   https://github.com/haotiansun14/AdaPlanner

Implementation:
    - Plan step format:  ``[Step N] <action>  ||  Expected: <outcome>``
    - The agent executes steps via tools or LLM reasoning, checks expected
      outcomes, and triggers refinement when expectations are violated.
    - Max refinement attempts capped to prevent loops.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """\
You are an expert planner that solves problems step by step.

Available tools:
{tool_descriptions}

Generate a PLAN as a numbered list.  Each step should specify:
  - What action to take (use a tool or reason).
  - What you expect the outcome to be.

Format EXACTLY like this:
[Step 1] Action: <describe action>  ||  Expected: <expected result>
[Step 2] Action: <describe action>  ||  Expected: <expected result>
...
[Step N] Action: Provide final answer  ||  Expected: <the answer>

Rules:
- To call a tool, write:  TOOL(<tool_name>, <input>)
- The last step must provide the final answer.
- Keep the plan concise (typically 3-6 steps).
"""

PLAN_USER = "Problem: {problem}\n\nGenerate your step-by-step plan:"

EXECUTE_SYSTEM = """\
You are an executor.  Given a problem, a plan step to execute, and previous \
observations, carry out the step and produce the result.

If the step says TOOL(name, input), call that tool.
Otherwise, reason through the step and give the result.

Output ONLY the result -- no extra commentary."""

EXECUTE_USER = """\
Problem: {problem}

Previous observations:
{history}

Current step to execute:
{step_action}

Result:"""

INPLAN_REFINE_SYSTEM = """\
You are a plan debugger.  A plan step produced an unexpected result.
Suggest a REVISED action for this step that will produce the expected outcome.

Output format:
Revised Action: <new action for this step>"""

INPLAN_REFINE_USER = """\
Problem: {problem}

Original step: {step_action}
Expected outcome: {expected}
Actual outcome: {actual}

Previous observations:
{history}

Suggest a revised action:"""

OUTPLAN_REFINE_SYSTEM = """\
You are a plan reviser.  The current plan has failed at a certain step.
Given the problem, the progress so far, and the failure, generate a NEW \
continuation plan from the failed step onward.

Use the same format:
[Step K] Action: <action>  ||  Expected: <expected result>
...
[Step N] Action: Provide final answer  ||  Expected: <answer>
"""

OUTPLAN_REFINE_USER = """\
Problem: {problem}

Progress so far:
{history}

Failed at step {step_num}:
  Action: {step_action}
  Expected: {expected}
  Actual: {actual}

Generate a revised continuation plan starting from step {step_num}:"""

CHECK_SYSTEM = """\
You are a strict verifier.  Given an expected outcome and an actual result, \
determine if they are consistent.

Output ONLY one word: PASS or FAIL"""

CHECK_USER = """\
Expected: {expected}
Actual: {actual}

Verdict:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"\[Step\s*(\d+)\]\s*Action:\s*(.*?)\s*\|\|\s*Expected:\s*(.*)",
    re.IGNORECASE,
)

_TOOL_RE = re.compile(r"TOOL\(\s*(\w+)\s*,\s*(.*?)\s*\)", re.IGNORECASE)


def _parse_plan(text: str) -> list[tuple[int, str, str]]:
    """Parse plan text into [(step_num, action, expected), ...]."""
    results = []
    for m in _STEP_RE.finditer(text):
        results.append((int(m.group(1)), m.group(2).strip(), m.group(3).strip()))
    return results


def _execute_tool_if_needed(action: str, tools) -> str | None:
    """If the action contains TOOL(...), execute it and return the result."""
    m = _TOOL_RE.search(action)
    if m:
        tool_name, tool_input = m.group(1).strip(), m.group(2).strip()
        try:
            tool = tools.get(tool_name)
            return tool.run(tool_input)
        except Exception as e:
            return f"Error: {e}"
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent
class AdaPlannerAgent(BaseAgent):
    paradigm_name = "AdaPlanner"
    paradigm_description = "Adaptive planning with in-plan and out-of-plan refinement from feedback."

    max_refine_attempts: int = 2  # per step

    def run(self, query: str) -> AgentResult:
        steps: list[ThinkStep] = []

        # ── 1. PLAN ──────────────────────────────────────────────
        tool_desc = self.tools.describe_all() if self.tools else "No tools available."
        plan_msgs = [
            {"role": "system", "content": PLAN_SYSTEM.format(tool_descriptions=tool_desc)},
            {"role": "user", "content": PLAN_USER.format(problem=query)},
        ]
        plan_response = self.llm.chat(plan_msgs, temperature=0.2)
        plan_steps = _parse_plan(plan_response)

        # Fallback if parsing fails
        if not plan_steps:
            steps.append(ThinkStep(type="plan", content=f"[Raw plan]\n{plan_response}"))
            steps.append(ThinkStep(type="answer", content=plan_response.strip()))
            return AgentResult(answer=plan_response.strip(), steps=steps)

        steps.append(ThinkStep(
            type="plan",
            content="\n".join(f"[Step {n}] {a}  ||  Expected: {e}" for n, a, e in plan_steps),
            metadata={"num_steps": len(plan_steps)},
        ))

        # ── 2. EXECUTE step by step ──────────────────────────────
        history_lines: list[str] = []
        final_answer = ""

        i = 0
        while i < len(plan_steps):
            step_num, action, expected = plan_steps[i]

            steps.append(ThinkStep(
                type="action",
                content=f"[Step {step_num}] {action}",
                metadata={"expected": expected},
            ))

            # Try tool execution first
            tool_result = _execute_tool_if_needed(action, self.tools)
            if tool_result is not None:
                actual = tool_result
            else:
                # LLM-based execution
                exec_msgs = [
                    {"role": "system", "content": EXECUTE_SYSTEM},
                    {"role": "user", "content": EXECUTE_USER.format(
                        problem=query,
                        history="\n".join(history_lines) if history_lines else "(none)",
                        step_action=action,
                    )},
                ]
                actual = self.llm.chat(exec_msgs, temperature=0.0).strip()

            steps.append(ThinkStep(
                type="observation",
                content=f"Result: {actual}",
                metadata={"step": step_num},
            ))

            # ── 3. CHECK assertion ────────────────────────────────
            check_msgs = [
                {"role": "system", "content": CHECK_SYSTEM},
                {"role": "user", "content": CHECK_USER.format(
                    expected=expected,
                    actual=actual,
                )},
            ]
            verdict = self.llm.chat(check_msgs, temperature=0.0).strip().upper()
            passed = "PASS" in verdict

            if passed:
                history_lines.append(f"[Step {step_num}] {action} -> {actual}")
                final_answer = actual  # last result becomes answer candidate
                i += 1
                continue

            # ── 4. IN-PLAN REFINEMENT ─────────────────────────────
            steps.append(ThinkStep(
                type="reflection",
                content=f"[Step {step_num}] FAILED assertion. Expected: {expected} | Got: {actual}",
            ))

            refined = False
            for attempt in range(1, self.max_refine_attempts + 1):
                refine_msgs = [
                    {"role": "system", "content": INPLAN_REFINE_SYSTEM},
                    {"role": "user", "content": INPLAN_REFINE_USER.format(
                        problem=query,
                        step_action=action,
                        expected=expected,
                        actual=actual,
                        history="\n".join(history_lines) if history_lines else "(none)",
                    )},
                ]
                refine_response = self.llm.chat(refine_msgs, temperature=0.3)
                # Extract revised action
                m = re.search(r"Revised Action:\s*(.*)", refine_response, re.DOTALL)
                new_action = m.group(1).strip() if m else refine_response.strip()

                steps.append(ThinkStep(
                    type="action",
                    content=f"[In-plan refine #{attempt}] {new_action}",
                    metadata={"step": step_num, "attempt": attempt},
                ))

                # Re-execute
                tool_result = _execute_tool_if_needed(new_action, self.tools)
                if tool_result is not None:
                    actual = tool_result
                else:
                    exec_msgs = [
                        {"role": "system", "content": EXECUTE_SYSTEM},
                        {"role": "user", "content": EXECUTE_USER.format(
                            problem=query,
                            history="\n".join(history_lines) if history_lines else "(none)",
                            step_action=new_action,
                        )},
                    ]
                    actual = self.llm.chat(exec_msgs, temperature=0.0).strip()

                steps.append(ThinkStep(type="observation", content=f"Result: {actual}"))

                # Re-check
                check_msgs[1] = {"role": "user", "content": CHECK_USER.format(expected=expected, actual=actual)}
                verdict = self.llm.chat(check_msgs, temperature=0.0).strip().upper()
                if "PASS" in verdict:
                    history_lines.append(f"[Step {step_num}] {new_action} -> {actual}")
                    final_answer = actual
                    refined = True
                    break

            if refined:
                i += 1
                continue

            # ── 5. OUT-OF-PLAN REFINEMENT ─────────────────────────
            steps.append(ThinkStep(
                type="reflection",
                content=f"[Step {step_num}] In-plan refinement exhausted. Triggering out-of-plan refinement.",
            ))

            outplan_msgs = [
                {"role": "system", "content": OUTPLAN_REFINE_SYSTEM},
                {"role": "user", "content": OUTPLAN_REFINE_USER.format(
                    problem=query,
                    history="\n".join(history_lines) if history_lines else "(none)",
                    step_num=step_num,
                    step_action=action,
                    expected=expected,
                    actual=actual,
                )},
            ]
            new_plan_text = self.llm.chat(outplan_msgs, temperature=0.3)
            new_steps = _parse_plan(new_plan_text)

            if new_steps:
                steps.append(ThinkStep(
                    type="plan",
                    content=f"[Revised plan from step {step_num}]\n"
                            + "\n".join(f"[Step {n}] {a}  ||  Expected: {e}" for n, a, e in new_steps),
                ))
                # Replace remaining plan steps
                plan_steps = plan_steps[:i] + new_steps
                # Don't increment i -- retry from current position with new step
            else:
                # Couldn't parse new plan, move on
                history_lines.append(f"[Step {step_num}] {action} -> {actual} (refined, unverified)")
                final_answer = actual
                i += 1

        # ── Final answer ──────────────────────────────────────────
        steps.append(ThinkStep(type="answer", content=final_answer))
        return AgentResult(answer=final_answer, steps=steps)

"""ReAct agent -- interleaved Reasoning and Acting.

Core idea (Yao et al., 2023):
    The model alternates between:
        Thought  -- reason about what to do next
        Action   -- call a tool
        Observation -- receive tool output
    This loop repeats until the model emits "Final Answer:".

Implementation:
    - System prompt describes available tools and the Thought/Action/Observation format.
    - Loop: send conversation to LLM, parse Thought + Action, execute tool,
      append Observation, repeat.
    - Cap at max_iterations to avoid runaway loops.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

SYSTEM_TEMPLATE = """\
You are a helpful assistant that can use tools to answer questions.

Available tools:
{tool_descriptions}

To use a tool, you MUST follow this EXACT format:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <input to the tool>

After you receive an Observation (the tool's output), continue with another Thought.

When you have enough information to answer, respond with:

Thought: I now have enough information to answer.
Final Answer: <your final answer>

IMPORTANT:
- Always start with a Thought.
- You can ONLY use the tools listed above.
- If a tool is not needed, go directly to Final Answer.
"""


def _parse_action(text: str) -> tuple[str | None, str | None]:
    """Extract Action and Action Input from LLM output."""
    action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text)
    action = action_match.group(1).strip() if action_match else None
    action_input = input_match.group(1).strip() if input_match else None
    return action, action_input


def _parse_final_answer(text: str) -> str | None:
    """Extract Final Answer if present."""
    match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_thought(text: str) -> str:
    """Extract the Thought line(s)."""
    match = re.search(r"Thought:\s*(.*?)(?=Action:|Final Answer:|$)", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


@register_agent
class ReActAgent(BaseAgent):
    paradigm_name = "ReAct"
    paradigm_description = "Interleave Thought / Action / Observation until solved."

    max_iterations: int = 6

    def run(self, query: str) -> AgentResult:
        # Build system prompt with tool descriptions
        tool_desc = self.tools.describe_all() if self.tools else "No tools available."
        system_msg = SYSTEM_TEMPLATE.format(tool_descriptions=tool_desc)

        messages: list[dict] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]
        steps: list[ThinkStep] = []

        for iteration in range(self.max_iterations):
            # ── call LLM ──────────────────────────────────────────
            response = self.llm.chat(messages, temperature=0.0, stop=["Observation:"])

            # ── check for final answer ────────────────────────────
            final = _parse_final_answer(response)
            if final is not None:
                thought = _parse_thought(response)
                if thought:
                    steps.append(ThinkStep(type="thought", content=thought))
                steps.append(ThinkStep(type="answer", content=final))
                return AgentResult(answer=final, steps=steps)

            # ── parse thought + action ────────────────────────────
            thought = _parse_thought(response)
            if thought:
                steps.append(ThinkStep(type="thought", content=thought))

            action, action_input = _parse_action(response)

            if action is None:
                # Model didn't follow format -- treat as final answer
                steps.append(ThinkStep(type="answer", content=response.strip()))
                return AgentResult(answer=response.strip(), steps=steps)

            steps.append(
                ThinkStep(
                    type="action",
                    content=f"{action}({action_input})",
                    metadata={"tool": action, "input": action_input},
                )
            )

            # ── execute tool ──────────────────────────────────────
            try:
                tool = self.tools.get(action)
                observation = tool.run(action_input or "")
            except KeyError:
                observation = f"Error: tool '{action}' not found. Available: {self.tools.describe_all()}"
            except Exception as exc:
                observation = f"Error executing {action}: {exc}"

            steps.append(
                ThinkStep(
                    type="observation",
                    content=observation,
                    metadata={"tool": action},
                )
            )

            # ── append to conversation for next turn ──────────────
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # ── max iterations reached ────────────────────────────────
        fallback = "I was unable to reach a final answer within the allowed steps."
        steps.append(ThinkStep(type="error", content=fallback))
        return AgentResult(answer=fallback, steps=steps)

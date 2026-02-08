"""Chain-of-Thought (CoT) agent.

Core idea (Wei et al., 2022):
    Prompt the model to "think step by step" before giving a final answer.
    This elicits intermediate reasoning steps that improve accuracy on
    complex tasks such as arithmetic, common-sense, and symbolic reasoning.

Implementation:
    - System prompt instructs step-by-step reasoning.
    - Single LLM call.
    - Parse the response to separate reasoning steps from the final answer.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

SYSTEM_PROMPT = """\
You are a helpful assistant that thinks step by step.

When answering a question:
1. Break down your reasoning into clear, numbered steps.
2. Show your work for each step.
3. After all reasoning steps, provide your final answer on a new line starting with "Final Answer:".

Example format:
Step 1: ...
Step 2: ...
Step 3: ...
Final Answer: ..."""


@register_agent
class CoTAgent(BaseAgent):
    paradigm_name = "CoT"
    paradigm_description = "Chain-of-Thought: think step by step before answering."

    def run(self, query: str) -> AgentResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response = self.llm.chat(messages, temperature=0.0)

        # ── parse reasoning steps and final answer ────────────────
        steps: list[ThinkStep] = []
        final_answer = response  # fallback: entire response

        # Try to split on "Final Answer:"
        parts = re.split(r"Final Answer:\s*", response, maxsplit=1, flags=re.IGNORECASE)
        reasoning_text = parts[0].strip()
        if len(parts) > 1:
            final_answer = parts[1].strip()

        # Extract individual steps (lines starting with "Step N:")
        step_pattern = re.compile(r"(Step\s+\d+[:.]\s*)(.*?)(?=Step\s+\d+[:.]\s*|$)", re.DOTALL)
        matches = step_pattern.findall(reasoning_text)

        if matches:
            for prefix, body in matches:
                content = (prefix + body).strip()
                steps.append(ThinkStep(type="thought", content=content))
        else:
            # Fallback: treat the whole reasoning text as one thought
            if reasoning_text:
                steps.append(ThinkStep(type="thought", content=reasoning_text))

        steps.append(ThinkStep(type="answer", content=final_answer))
        return AgentResult(answer=final_answer, steps=steps)

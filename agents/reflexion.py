"""Reflexion agent -- verbal reinforcement learning through self-reflection.

Core idea (Shinn et al., 2023 -- NeurIPS):
    Reflexion reinforces LLM agents not through weight updates, but through
    **verbal reflection**.  After each trial the agent:
        1. Attempts to solve the task (Act -- using CoT or ReAct internally).
        2. Evaluates whether the attempt succeeded.
        3. If it failed, generates a **reflection** -- a natural-language
           diagnosis of what went wrong and how to do better.
        4. The reflection is stored in an episodic memory buffer.
        5. On the next trial the reflections are prepended to the prompt,
           giving the agent "verbal experience" to improve.

    Ref paper:  https://arxiv.org/abs/2303.11366
    Ref impl:   https://github.com/noahshinn/reflexion

Implementation (single-turn adaptation):
    Since we are in single-turn mode, we simulate the multi-trial loop
    *within one call to run()*:
        Trial 1  →  Act (CoT)  →  self-evaluate  →  reflect
        Trial 2  →  Act (CoT + reflections)  →  self-evaluate  →  reflect
        ...
        Trial N  →  Act (CoT + all reflections)  →  return best answer

    The self-evaluation and reflection are done by separate LLM calls, as
    in the original paper.  Max trials is capped (default 3).
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ── prompts ───────────────────────────────────────────────────────

COT_SYSTEM = """\
You are a helpful assistant that thinks step by step.

When answering a question:
1. Break down your reasoning into clear, numbered steps.
2. Show your work for each step.
3. After all reasoning steps, provide your final answer on a new line starting with "Final Answer:".

Example format:
Step 1: ...
Step 2: ...
Final Answer: ..."""

EVALUATE_PROMPT = """\
You are an evaluator. Given a question and a proposed answer, assess whether \
the answer is likely correct and complete.

Question: {question}

Proposed Answer: {answer}

Respond with EXACTLY one of:
- "CORRECT" if the answer is likely correct and complete.
- "INCORRECT: <brief reason>" if the answer is wrong or incomplete.
"""

REFLECT_PROMPT = """\
You are a reflection assistant. A previous attempt to answer a question failed. \
Analyze the reasoning trace and identify what went wrong and how to improve.

Question: {question}

Previous reasoning and answer:
{scratchpad}

Evaluation feedback: {evaluation}

Write a concise reflection (2-4 sentences) on:
1. What specifically went wrong in the reasoning.
2. What strategy should be used in the next attempt.

Reflection:"""

REFLEXION_HEADER = "You have attempted this question before. " \
    "Here are reflections on your previous attempts:\n\n"


# ── helpers ───────────────────────────────────────────────────────

def _extract_final_answer(text: str) -> str:
    """Extract text after 'Final Answer:' or return the full text."""
    parts = re.split(r"Final Answer:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    return parts[1].strip() if len(parts) > 1 else text.strip()


# ── agent ─────────────────────────────────────────────────────────

@register_agent
class ReflexionAgent(BaseAgent):
    paradigm_name = "Reflexion"
    paradigm_description = "Self-reflect on failures and retry with accumulated verbal insight."

    max_trials: int = 3

    def run(self, query: str) -> AgentResult:
        reflections: list[str] = []
        steps: list[ThinkStep] = []
        best_answer: str = ""

        for trial in range(1, self.max_trials + 1):
            steps.append(ThinkStep(
                type="plan",
                content=f"--- Trial {trial} / {self.max_trials} ---",
                metadata={"trial": trial},
            ))

            # ── 1. Act: generate answer using CoT + reflections ──
            system_msg = COT_SYSTEM
            if reflections:
                reflection_block = REFLEXION_HEADER
                for i, r in enumerate(reflections, 1):
                    reflection_block += f"- Attempt {i}: {r}\n"
                reflection_block += "\nUse these reflections to improve your answer.\n\n"
                system_msg = reflection_block + system_msg

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ]
            response = self.llm.chat(messages, temperature=0.0)
            best_answer = _extract_final_answer(response)

            steps.append(ThinkStep(type="thought", content=response))

            # ── 2. Evaluate: ask LLM to assess correctness ───────
            eval_prompt = EVALUATE_PROMPT.format(
                question=query,
                answer=best_answer,
            )
            evaluation = self.llm.chat(
                [{"role": "user", "content": eval_prompt}],
                temperature=0.0,
            ).strip()

            steps.append(ThinkStep(
                type="observation",
                content=f"Self-evaluation: {evaluation}",
                metadata={"trial": trial},
            ))

            # If the evaluator says CORRECT, return early
            if evaluation.upper().startswith("CORRECT"):
                steps.append(ThinkStep(type="answer", content=best_answer))
                return AgentResult(answer=best_answer, steps=steps)

            # ── 3. Reflect: diagnose failure ─────────────────────
            if trial < self.max_trials:
                reflect_prompt = REFLECT_PROMPT.format(
                    question=query,
                    scratchpad=response,
                    evaluation=evaluation,
                )
                reflection = self.llm.chat(
                    [{"role": "user", "content": reflect_prompt}],
                    temperature=0.0,
                ).strip()

                reflections.append(reflection)
                steps.append(ThinkStep(
                    type="reflection",
                    content=reflection,
                    metadata={"trial": trial},
                ))

        # ── exhausted all trials, return best answer ─────────────
        steps.append(ThinkStep(type="answer", content=best_answer))
        return AgentResult(answer=best_answer, steps=steps)

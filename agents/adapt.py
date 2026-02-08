"""ADaPT agent -- As-Needed Decomposition and Planning with LLMs.

Core idea (Prasad et al., 2023 -- NAACL):
    A recursive controller that decomposes tasks *only when needed*.
    Three components work together:

        - **Executor** -- attempts to solve a (sub-)task directly.
        - **Planner**  -- if the executor fails or is uncertain,
                          decomposes the task into smaller sub-tasks
                          with a logical operator (AND / OR).
        - **Controller** -- a pre-determined recursive algorithm that
                           orchestrates executor and planner, propagating
                           information between sub-tasks.

    Key insight: unlike plan-then-execute approaches that decompose
    everything upfront, ADaPT decomposes *adaptively* -- only when the
    LLM cannot handle a sub-task at the current granularity, it recurses
    deeper.  A ``max_depth`` limit prevents infinite recursion.

    ADaPT outperforms baselines by 28-33% on ALFWorld, WebShop, TextCraft.

    Ref paper:  https://arxiv.org/abs/2311.05772
    Ref impl:   https://github.com/archiki/ADaPT

Implementation:
    Adapted for general QA (no interactive environment):
    - Executor:   LLM tries to answer directly, then self-assesses
                  confidence (HIGH / LOW).
    - Planner:    LLM decomposes into numbered sub-tasks with AND/OR logic.
    - Controller: Recursive ``_solve()`` method.  On LOW confidence,
                  decomposes and recurses.  Sub-answers are combined
                  according to the logical operator.
    - Information from solved sub-tasks is propagated to subsequent ones.
    - All recursive steps logged as ThinkStep for UI visualization.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXECUTE_SYSTEM = """\
You are a helpful assistant.  Answer the given question or sub-question \
directly and concisely.

After your answer, assess your own confidence on a NEW line:
Confidence: HIGH   (if you are sure the answer is correct)
Confidence: LOW    (if you are uncertain or the question is too complex)
"""

EXECUTE_USER = """\
{context}Question: {question}

Answer:"""

PLAN_SYSTEM = """\
You are an expert task decomposer.  A question was too complex to answer \
directly.  Break it into smaller, simpler sub-questions whose answers can \
be combined to answer the original.

Rules:
- Output 2-4 sub-questions, numbered.
- After the sub-questions, output the logical operator:
    Logic: AND   (all sub-answers are needed)
    Logic: OR    (any one sub-answer suffices)
- Each sub-question should be self-contained and simpler than the original.

Example:
1. What is X?
2. What is Y?
3. How do X and Y relate to Z?
Logic: AND
"""

PLAN_USER = """\
Original question: {question}

{context}Decompose into simpler sub-questions:"""

COMBINE_SYSTEM = """\
You are a synthesiser.  Given the original question and answers to its \
sub-questions, combine them into a single final answer.

Be concise.  Output ONLY the combined answer."""

COMBINE_USER = """\
Original question: {question}

Sub-questions and their answers:
{sub_qa_text}

Combined answer:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_confidence(text: str) -> tuple[str, str]:
    """Split answer text and confidence level."""
    m = re.search(r"Confidence:\s*(HIGH|LOW)", text, re.IGNORECASE)
    confidence = m.group(1).upper() if m else "HIGH"
    answer = re.sub(r"Confidence:\s*(HIGH|LOW)\s*$", "", text, flags=re.IGNORECASE).strip()
    return answer, confidence


def _parse_subtasks(text: str) -> tuple[list[str], str]:
    """Parse numbered sub-questions and logic operator."""
    pattern = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)
    questions = [m.strip() for m in pattern.findall(text)]
    # Parse logic
    logic_m = re.search(r"Logic:\s*(AND|OR)", text, re.IGNORECASE)
    logic = logic_m.group(1).upper() if logic_m else "AND"
    # Fallback
    if not questions:
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip() and not ln.strip().lower().startswith("logic")]
        questions = [re.sub(r"^\d+[\.\)]\s*", "", ln) for ln in lines]
    return questions, logic


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent
class ADaPTAgent(BaseAgent):
    paradigm_name = "ADaPT"
    paradigm_description = "As-needed recursive decomposition: solve directly, or decompose and recurse."

    max_depth: int = 3

    def run(self, query: str) -> AgentResult:
        steps: list[ThinkStep] = []
        answer = self._solve(query, depth=1, steps=steps, context="")
        steps.append(ThinkStep(type="answer", content=answer))
        return AgentResult(answer=answer, steps=steps)

    # ── recursive controller ──────────────────────────────────────

    def _solve(
        self,
        question: str,
        depth: int,
        steps: list[ThinkStep],
        context: str,
    ) -> str:
        indent = "  " * (depth - 1)

        # ── 1. EXECUTE: try to answer directly ────────────────────
        steps.append(ThinkStep(
            type="action",
            content=f"{indent}[Depth {depth}] Executor: attempting '{question[:100]}...'",
            metadata={"depth": depth},
        ))

        ctx_block = f"Context from previous steps:\n{context}\n\n" if context else ""
        exec_msgs = [
            {"role": "system", "content": EXECUTE_SYSTEM},
            {"role": "user", "content": EXECUTE_USER.format(
                question=question,
                context=ctx_block,
            )},
        ]
        exec_response = self.llm.chat(exec_msgs, temperature=0.2)
        answer, confidence = _parse_confidence(exec_response)

        steps.append(ThinkStep(
            type="observation",
            content=f"{indent}[Depth {depth}] Answer: {answer[:200]}  |  Confidence: {confidence}",
            metadata={"depth": depth, "confidence": confidence},
        ))

        # ── If confident or at max depth, return ──────────────────
        if confidence == "HIGH" or depth >= self.max_depth:
            if depth >= self.max_depth and confidence == "LOW":
                steps.append(ThinkStep(
                    type="reflection",
                    content=f"{indent}[Depth {depth}] Max depth reached, returning best-effort answer.",
                ))
            return answer

        # ── 2. PLAN: decompose into sub-tasks ─────────────────────
        steps.append(ThinkStep(
            type="reflection",
            content=f"{indent}[Depth {depth}] LOW confidence -- triggering decomposition.",
        ))

        plan_msgs = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": PLAN_USER.format(
                question=question,
                context=ctx_block,
            )},
        ]
        plan_response = self.llm.chat(plan_msgs, temperature=0.3)
        sub_questions, logic = _parse_subtasks(plan_response)

        if not sub_questions:
            # Couldn't decompose, return the direct answer
            return answer

        steps.append(ThinkStep(
            type="plan",
            content=f"{indent}[Depth {depth}] Decomposed into {len(sub_questions)} sub-tasks (logic: {logic}):\n"
                    + "\n".join(f"{indent}  {i}. {q}" for i, q in enumerate(sub_questions, 1)),
            metadata={"depth": depth, "logic": logic, "sub_questions": sub_questions},
        ))

        # ── 3. CONTROLLER: recurse into sub-tasks ────────────────
        sub_answers: list[tuple[str, str]] = []   # (question, answer)
        propagated_context = context

        for i, sub_q in enumerate(sub_questions, 1):
            sub_answer = self._solve(
                question=sub_q,
                depth=depth + 1,
                steps=steps,
                context=propagated_context,
            )
            sub_answers.append((sub_q, sub_answer))

            # OR logic: return on first success
            if logic == "OR":
                steps.append(ThinkStep(
                    type="thought",
                    content=f"{indent}[Depth {depth}] OR logic: accepting first sub-answer.",
                ))
                return sub_answer

            # AND logic: propagate information to next sub-task
            propagated_context += f"\n- {sub_q} -> {sub_answer}"

        # ── 4. COMBINE sub-answers ────────────────────────────────
        sub_qa_text = "\n".join(
            f"{i}. Q: {q}\n   A: {a}" for i, (q, a) in enumerate(sub_answers, 1)
        )

        steps.append(ThinkStep(
            type="plan",
            content=f"{indent}[Depth {depth}] Combining {len(sub_answers)} sub-answers...",
        ))

        combine_msgs = [
            {"role": "system", "content": COMBINE_SYSTEM},
            {"role": "user", "content": COMBINE_USER.format(
                question=question,
                sub_qa_text=sub_qa_text,
            )},
        ]
        combined = self.llm.chat(combine_msgs, temperature=0.0).strip()

        steps.append(ThinkStep(
            type="thought",
            content=f"{indent}[Depth {depth}] Combined answer: {combined[:300]}",
            metadata={"depth": depth},
        ))

        return combined

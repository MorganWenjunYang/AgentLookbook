"""Graph of Thoughts (GoT) agent -- graph-based reasoning with aggregation & refinement.

Core idea (Besta et al., 2024 -- AAAI):
    Extends Tree of Thoughts by modelling LLM reasoning as an *arbitrary
    directed graph* rather than a tree.  Thoughts are vertices; edges
    represent dependencies.  The key new operations beyond ToT are:

        - **Aggregate** -- merge several partial solutions into one
          (e.g. combine two sorted sub-lists into a merged sorted list).
        - **Refine**    -- iteratively improve a thought based on LLM feedback.

    These allow decomposition strategies (split problem -> solve parts ->
    merge) that a tree cannot express.

    GoT increases quality by 62 %% over ToT on sorting while reducing costs
    by over 31 %%.

    Ref paper:  https://arxiv.org/abs/2308.09687
    Ref impl:   https://github.com/spcl/graph-of-thoughts

Implementation:
    Our general-purpose GoT for QA problems follows this graph pattern:

        [Query]
           |
      ┌────┼────┐          GENERATE  -- decompose into sub-problems
      v    v    v
     S1   S2   S3          SOLVE     -- solve each sub-problem independently
      \    |    /
       \   |   /
        v  v  v
       [Aggregate]          AGGREGATE -- merge sub-answers into one
           |
        [Refine]            REFINE    -- self-critique and improve
           |
        [Answer]

    Every operation is recorded as a ThinkStep for the UI.
"""

from __future__ import annotations

import re

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """\
You are an expert problem decomposer.  Given a problem, break it down into \
{n_branches} smaller, independent sub-problems whose answers can later be \
combined to solve the original.

Rules:
- Each sub-problem should be self-contained and answerable on its own.
- Output each sub-problem on its own line, prefixed by its number:
    1. <sub-problem>
    2. <sub-problem>
    ...
- If the problem is already simple enough, just restate it as sub-problem 1.
"""

DECOMPOSE_USER = """\
Problem:
{problem}

Break it into {n_branches} sub-problems:"""

SOLVE_SYSTEM = """\
You are a helpful assistant.  Solve the given sub-problem step by step.
Be concise but thorough.  End with:
Sub-Answer: <your answer to this sub-problem>"""

SOLVE_USER = """\
Original problem (for context):
{problem}

Sub-problem to solve:
{sub_problem}

Solve it:"""

AGGREGATE_SYSTEM = """\
You are an expert synthesiser.  You are given the original problem and several \
sub-answers to its decomposed parts.  Combine them into a single, coherent \
answer to the original problem.

Output format:
Merged Reasoning: <explain how the sub-answers combine>
Merged Answer: <the combined answer>"""

AGGREGATE_USER = """\
Original problem:
{problem}

Sub-answers:
{sub_answers_text}

Aggregate into a single answer:"""

REFINE_SYSTEM = """\
You are a meticulous reviewer.  Given a problem and a draft answer, \
critically evaluate it: check for errors, missing steps, or incorrect logic.
If you find issues, produce an improved answer.  If the answer is already \
correct, keep it as-is.

Output format:
Critique: <what is wrong, or "looks correct">
Refined Answer: <improved answer, or same if already correct>"""

REFINE_USER = """\
Problem:
{problem}

Draft answer:
{draft_answer}

Review and refine:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_numbered_items(text: str, n: int) -> list[str]:
    """Extract numbered items from LLM output."""
    pattern = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)
    matches = pattern.findall(text)
    if not matches:
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        matches = [re.sub(r"^\d+[\.\)]\s*", "", ln) for ln in lines]
    return matches[:n] if len(matches) >= n else matches if matches else [text.strip()]


def _extract_after(text: str, label: str) -> str:
    """Extract text after a label like 'Sub-Answer:' or 'Refined Answer:'."""
    pattern = re.compile(rf"{label}\s*(.*)", re.DOTALL | re.IGNORECASE)
    m = pattern.search(text)
    return m.group(1).strip() if m else text.strip()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent
class GoTAgent(BaseAgent):
    paradigm_name = "GoT"
    paradigm_description = "Graph of Thoughts: decompose, solve sub-problems, aggregate, and refine."

    n_branches: int = 3         # number of sub-problems to decompose into
    n_refine_iters: int = 1     # refinement iterations

    def run(self, query: str) -> AgentResult:
        steps: list[ThinkStep] = []

        # ── 1. DECOMPOSE (Generate) ──────────────────────────────
        steps.append(ThinkStep(
            type="plan",
            content=f"[Decompose] Breaking the problem into {self.n_branches} sub-problems...",
        ))

        decompose_msgs = [
            {"role": "system", "content": DECOMPOSE_SYSTEM.format(n_branches=self.n_branches)},
            {"role": "user", "content": DECOMPOSE_USER.format(
                problem=query,
                n_branches=self.n_branches,
            )},
        ]
        decompose_response = self.llm.chat(decompose_msgs, temperature=0.7)
        sub_problems = _parse_numbered_items(decompose_response, self.n_branches)

        for i, sp in enumerate(sub_problems, 1):
            steps.append(ThinkStep(
                type="thought",
                content=f"Sub-problem {i}: {sp}",
                metadata={"sub_problem_index": i},
            ))

        # ── 2. SOLVE each sub-problem independently ──────────────
        sub_answers: list[str] = []
        for i, sp in enumerate(sub_problems, 1):
            solve_msgs = [
                {"role": "system", "content": SOLVE_SYSTEM},
                {"role": "user", "content": SOLVE_USER.format(
                    problem=query,
                    sub_problem=sp,
                )},
            ]
            solve_response = self.llm.chat(solve_msgs, temperature=0.2)
            sub_answer = _extract_after(solve_response, "Sub-Answer:")
            sub_answers.append(sub_answer)

            steps.append(ThinkStep(
                type="observation",
                content=f"[Solve] Sub-problem {i} -> {sub_answer[:200]}",
                metadata={"sub_problem": sp, "sub_answer": sub_answer},
            ))

        # ── 3. AGGREGATE -- merge sub-answers ────────────────────
        steps.append(ThinkStep(
            type="plan",
            content=f"[Aggregate] Merging {len(sub_answers)} sub-answers into a unified answer...",
        ))

        sub_answers_text = "\n".join(
            f"{i}. {sa}" for i, sa in enumerate(sub_answers, 1)
        )
        agg_msgs = [
            {"role": "system", "content": AGGREGATE_SYSTEM},
            {"role": "user", "content": AGGREGATE_USER.format(
                problem=query,
                sub_answers_text=sub_answers_text,
            )},
        ]
        agg_response = self.llm.chat(agg_msgs, temperature=0.2)
        merged_answer = _extract_after(agg_response, "Merged Answer:")

        steps.append(ThinkStep(
            type="thought",
            content=f"[Aggregate result] {merged_answer[:300]}",
            metadata={"full_merge": agg_response},
        ))

        # ── 4. REFINE -- self-critique and improve ───────────────
        draft = merged_answer
        for r in range(1, self.n_refine_iters + 1):
            steps.append(ThinkStep(
                type="plan",
                content=f"[Refine] Iteration {r}/{self.n_refine_iters}...",
            ))

            refine_msgs = [
                {"role": "system", "content": REFINE_SYSTEM},
                {"role": "user", "content": REFINE_USER.format(
                    problem=query,
                    draft_answer=draft,
                )},
            ]
            refine_response = self.llm.chat(refine_msgs, temperature=0.2)
            critique = _extract_after(refine_response, "Critique:")
            refined = _extract_after(refine_response, "Refined Answer:")

            steps.append(ThinkStep(
                type="reflection",
                content=f"[Critique] {critique[:300]}",
            ))
            steps.append(ThinkStep(
                type="thought",
                content=f"[Refined] {refined[:300]}",
                metadata={"iteration": r},
            ))

            draft = refined

        # ── Final answer ──────────────────────────────────────────
        steps.append(ThinkStep(type="answer", content=draft))
        return AgentResult(answer=draft, steps=steps)

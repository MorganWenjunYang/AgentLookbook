"""Tree of Thoughts (ToT) agent -- deliberate problem-solving via BFS.

Core idea (Yao et al., 2023 -- NeurIPS):
    Generalises Chain-of-Thought by exploring *multiple* reasoning paths
    (thoughts) at each step, evaluating them, and keeping only the most
    promising ones.  This turns LLM reasoning into a tree search:

        1. **Generate** -- propose several candidate next-steps (thoughts).
        2. **Evaluate** -- score each candidate with a second LLM call.
        3. **Select**   -- keep the top-k candidates.
        4. Repeat for a configurable number of depth levels.

    The search strategy implemented here is BFS (Breadth-First Search)
    with a beam width of ``n_select``, following the official reference
    implementation at ``princeton-nlp/tree-of-thought-llm``.

    Ref paper:  https://arxiv.org/abs/2305.10601
    Ref impl:   https://github.com/princeton-nlp/tree-of-thought-llm

Implementation:
    - A *generate* prompt asks the LLM for ``n_generate`` distinct
      next-step proposals given the current partial solution.
    - An *evaluate* prompt asks the LLM to rate each candidate on a
      scale of 1-10 (averaged over ``n_eval_samples``).
    - After all depth levels, the best surviving candidate is used
      with a final "synthesise answer" call.
    - Every generate / evaluate / select action is logged as a ThinkStep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

GENERATE_SYSTEM = """\
You are an expert problem solver.  You will be given a problem and a partial \
reasoning chain.  Your task is to propose {n_generate} distinct possible NEXT \
STEPS that continue the reasoning.

Rules:
- Each proposal should be a single coherent reasoning step (1-3 sentences).
- Proposals should be *diverse* -- explore different angles.
- Output each proposal on its own line, prefixed with its number:
    1. <proposal>
    2. <proposal>
    ...
- Output EXACTLY {n_generate} proposals, no more, no less.
"""

GENERATE_USER = """\
Problem:
{problem}

Reasoning so far:
{reasoning_so_far}

Propose {n_generate} distinct next reasoning steps:"""

EVALUATE_SYSTEM = """\
You are a critical evaluator.  Given a problem and a candidate partial \
reasoning chain, rate how promising this chain is for reaching the correct \
answer.

Output ONLY a single integer from 1 to 10 (10 = very promising, 1 = dead end).
Do NOT output any other text -- just the number."""

EVALUATE_USER = """\
Problem:
{problem}

Candidate reasoning:
{candidate}

Score (1-10):"""

SYNTHESISE_SYSTEM = """\
You are a helpful assistant.  Given a problem and the best reasoning path \
found through deliberate exploration, provide the final answer.

Format:
Reasoning:
<your consolidated reasoning>

Final Answer: <concise answer>"""

SYNTHESISE_USER = """\
Problem:
{problem}

Best reasoning path:
{best_path}

Provide the final answer:"""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    """A partial reasoning path being explored."""
    reasoning: str = ""
    score: float = 0.0


def _parse_proposals(text: str, n: int) -> list[str]:
    """Extract numbered proposals from LLM output."""
    pattern = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)
    matches = pattern.findall(text)
    # If parsing fails, split by newlines and take non-empty lines
    if len(matches) < n:
        lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
        # strip leading number prefixes
        cleaned = []
        for ln in lines:
            cleaned.append(re.sub(r"^\d+[\.\)]\s*", "", ln))
        matches = cleaned
    return matches[:n] if len(matches) >= n else matches if matches else [text.strip()]


def _parse_score(text: str) -> float:
    """Extract a numeric score from LLM output."""
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", text.strip())
    return float(m.group(1)) if m else 5.0  # default middle score


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@register_agent
class ToTAgent(BaseAgent):
    paradigm_name = "ToT"
    paradigm_description = "Tree of Thoughts: explore multiple reasoning paths via BFS, evaluate, and select the best."

    # Search hyper-parameters (sensible defaults for general QA)
    max_depth: int = 3          # number of BFS levels
    n_generate: int = 3         # proposals per candidate per level
    n_select: int = 2           # beam width -- keep top-k per level
    n_eval_samples: int = 1     # LLM calls per evaluation (averaged)

    def run(self, query: str) -> AgentResult:
        steps: list[ThinkStep] = []

        # Initial beam: one empty candidate
        beam: list[_Candidate] = [_Candidate(reasoning="(start)")]

        # ── BFS loop ──────────────────────────────────────────────
        for depth in range(1, self.max_depth + 1):
            steps.append(ThinkStep(
                type="plan",
                content=f"=== Depth {depth}/{self.max_depth}  |  beam size = {len(beam)} ===",
            ))

            all_candidates: list[_Candidate] = []

            # 1. GENERATE -- expand each beam candidate
            for parent in beam:
                gen_msgs = [
                    {"role": "system", "content": GENERATE_SYSTEM.format(n_generate=self.n_generate)},
                    {"role": "user", "content": GENERATE_USER.format(
                        problem=query,
                        reasoning_so_far=parent.reasoning,
                        n_generate=self.n_generate,
                    )},
                ]
                gen_response = self.llm.chat(gen_msgs, temperature=0.8)
                proposals = _parse_proposals(gen_response, self.n_generate)

                steps.append(ThinkStep(
                    type="thought",
                    content=f"[Generate] from '{parent.reasoning[:60]}...' -> {len(proposals)} proposals",
                    metadata={"proposals": proposals, "depth": depth},
                ))

                for prop in proposals:
                    new_reasoning = f"{parent.reasoning}\nStep {depth}: {prop}" if parent.reasoning != "(start)" else f"Step {depth}: {prop}"
                    all_candidates.append(_Candidate(reasoning=new_reasoning))

            # 2. EVALUATE -- score every candidate
            for cand in all_candidates:
                scores: list[float] = []
                for _ in range(self.n_eval_samples):
                    eval_msgs = [
                        {"role": "system", "content": EVALUATE_SYSTEM},
                        {"role": "user", "content": EVALUATE_USER.format(
                            problem=query,
                            candidate=cand.reasoning,
                        )},
                    ]
                    eval_response = self.llm.chat(eval_msgs, temperature=0.3)
                    scores.append(_parse_score(eval_response))
                cand.score = sum(scores) / len(scores)

            steps.append(ThinkStep(
                type="observation",
                content=f"[Evaluate] scored {len(all_candidates)} candidates  |  "
                        f"scores: {[round(c.score, 1) for c in sorted(all_candidates, key=lambda c: c.score, reverse=True)]}",
                metadata={"depth": depth},
            ))

            # 3. SELECT -- keep top-k
            all_candidates.sort(key=lambda c: c.score, reverse=True)
            beam = all_candidates[:self.n_select]

            steps.append(ThinkStep(
                type="thought",
                content=f"[Select] kept top-{self.n_select}: scores = {[round(c.score, 1) for c in beam]}",
                metadata={"depth": depth, "selected": [c.reasoning for c in beam]},
            ))

        # ── Synthesise final answer from the best path ────────────
        best = beam[0]
        steps.append(ThinkStep(
            type="thought",
            content=f"[Best path] (score {best.score:.1f}):\n{best.reasoning}",
        ))

        synth_msgs = [
            {"role": "system", "content": SYNTHESISE_SYSTEM},
            {"role": "user", "content": SYNTHESISE_USER.format(
                problem=query,
                best_path=best.reasoning,
            )},
        ]
        synth_response = self.llm.chat(synth_msgs, temperature=0.0)

        # Extract final answer
        answer = synth_response
        fa_match = re.search(r"Final Answer:\s*(.*)", synth_response, re.DOTALL)
        if fa_match:
            answer = fa_match.group(1).strip()

        steps.append(ThinkStep(type="answer", content=answer))
        return AgentResult(answer=answer, steps=steps)

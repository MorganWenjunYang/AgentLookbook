"""Vanilla agent -- direct LLM call with no special prompting (baseline)."""

from __future__ import annotations

from agents.base import AgentResult, BaseAgent, ThinkStep
from agents.registry import register_agent


@register_agent
class VanillaAgent(BaseAgent):
    paradigm_name = "Vanilla"
    paradigm_description = "Direct LLM call with no special prompting. Baseline for comparison."

    def run(self, query: str) -> AgentResult:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]

        response = self.llm.chat(messages)

        steps = [
            ThinkStep(type="response", content=response),
        ]
        return AgentResult(answer=response, steps=steps)

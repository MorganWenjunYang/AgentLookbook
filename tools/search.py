"""Mock search tool -- returns canned results for demonstration."""

from __future__ import annotations

from tools.base import Tool

# A small set of canned responses for common demo queries
_CANNED: dict[str, str] = {
    "python": "Python is a high-level programming language created by Guido van Rossum in 1991.",
    "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
    "chain of thought": "Chain-of-Thought (CoT) prompting asks a model to show intermediate reasoning steps.",
    "react": "ReAct interleaves reasoning traces and actions, allowing LLMs to interact with tools.",
    "agent": "An AI agent is a system that perceives its environment and takes actions to achieve goals.",
}


class WikiSearchTool(Tool):
    name = "WikiSearch"
    description = (
        "Searches a knowledge base and returns a short summary. "
        "Input should be a search query string."
    )

    def run(self, input_text: str) -> str:
        query = input_text.strip().lower()
        # Try exact or substring match
        for key, value in _CANNED.items():
            if key in query or query in key:
                return value
        return (
            f"No results found for '{input_text}'. "
            "This is a mock search tool with limited knowledge."
        )

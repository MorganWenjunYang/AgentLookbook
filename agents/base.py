"""Base agent class and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from llm.base import LLMClient
from tools.base import ToolRegistry


@dataclass
class ThinkStep:
    """One intermediate step in an agent's reasoning trace.

    Used by the UI to visualize the agent's thought process.
    """

    type: str           # "thought", "action", "observation", "reflection", "plan", etc.
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    """The final output of an agent run."""

    answer: str
    steps: list[ThinkStep] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agent paradigms.

    Subclasses MUST set:
        paradigm_name        -- short identifier, e.g. "ReAct"
        paradigm_description -- one-line explanation for the UI

    And implement:
        run(query) -> AgentResult
    """

    paradigm_name: str = ""
    paradigm_description: str = ""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()

    @abstractmethod
    def run(self, query: str) -> AgentResult:
        """Execute the agent paradigm on the given query."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(paradigm={self.paradigm_name!r})"

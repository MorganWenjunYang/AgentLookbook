"""Agent registry -- auto-discovers paradigms for the UI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.base import BaseAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Class decorator that registers an agent paradigm.

    Usage:
        @register_agent
        class MyAgent(BaseAgent):
            paradigm_name = "MyParadigm"
            ...
    """
    AGENT_REGISTRY[cls.paradigm_name] = cls
    return cls

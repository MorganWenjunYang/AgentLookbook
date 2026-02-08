"""Tool abstraction and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Tool(ABC):
    """A tool that an agent can invoke."""

    name: str
    description: str

    @abstractmethod
    def run(self, input_text: str) -> str:
        """Execute the tool and return a string result."""


class ToolRegistry:
    """Holds available tools and produces descriptions for prompt injection."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {list(self._tools)}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def describe_all(self) -> str:
        """Return a formatted description of all tools for prompt injection."""
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

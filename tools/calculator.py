"""Simple calculator tool -- evaluates basic math expressions."""

from __future__ import annotations

from tools.base import Tool


class CalculatorTool(Tool):
    name = "Calculator"
    description = (
        "Evaluates a mathematical expression. "
        "Input should be a valid Python math expression, e.g. '2 + 3 * 4'."
    )

    def run(self, input_text: str) -> str:
        try:
            # Only allow safe math operations
            allowed = set("0123456789+-*/.() %")
            expr = input_text.strip()
            if not all(ch in allowed for ch in expr):
                return f"Error: expression contains invalid characters: {expr}"
            result = eval(expr)  # noqa: S307  -- intentionally limited
            return str(result)
        except Exception as exc:
            return f"Error: {exc}"

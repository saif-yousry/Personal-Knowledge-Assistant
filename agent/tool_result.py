"""
Every tool function, no matter what it does internally, must return one
of these.
This is the contract tools.py implementers build against —
the controller never inspects a tool's raw return value, only this.
"""

from __future__ import annotations

from pydantic import BaseModel


class ToolResult(BaseModel):
    """Represents a tool result."""
    tool_name: str
    success: bool
    data: dict | None = None
    error: str | None = None

    def to_observation_text(self) -> str:
        """How this result gets serialized back into the message history
        for the next LLM call to read. Keep it short and plain — this
        text becomes part of the prompt on every subsequent turn."""

        if self.success:
            return f"[{self.tool_name}] result: {self.data}"
        return f"[{self.tool_name}] error: {self.error}"

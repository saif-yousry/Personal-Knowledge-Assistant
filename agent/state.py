"""
Conversation state for the agentic LLM loop.

Tracks the message history in the OpenAI messages format
(list of dicts with "role" and "content" keys).
"""

from __future__ import annotations


class AgentState:
    """Holds the conversation history and current user input."""

    def __init__(self, user_input: str, messages: list[dict] | None = None) -> None:
        self.user_input = user_input
        self.messages: list[dict] = list(messages) if messages else []
        self.messages.append({"role": "user", "content": user_input})

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool result to the conversation history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

"""
This is the piece that implements your core idea: the LLM only ever
names a tool, the code decides what running that tool means.

TOOL_REGISTRY maps a tool name (string) to a callable. Adding a tool is
a one-line registration here — no if/elif chain to extend, no touching
controller.py. Each tool function's signature is the contract teammate B
builds every tool against:

    def some_tool(arguments: dict) -> ToolResult:
        ...

Register new tools at the bottom of this file, or import them from
testing_tools.py / a future tools/weather.py etc. and add them to the dict.
"""

from typing import Callable
from agent.tool_result import ToolResult
# Reason: Imported `send_email` alongside `search` so it can be registered in TOOL_REGISTRY.
from agent.tools import search, send_email

ToolFn = Callable[[dict], ToolResult]

# Reason: Added "send_email" to TOOL_REGISTRY. When the LLM decides to call `send_email`,
# `execute_tool` performs an O(1) dictionary lookup to invoke the `send_email` tool.
TOOL_REGISTRY: dict[str, ToolFn] = {
    "search": search,
    "send_email": send_email,
}


def execute_tool(
    name: str,
    arguments: dict,
    *,
    user_id: int | None = None,
) -> ToolResult:

    """This is the actual 'loop that checks the tool's name' — a dict
    lookup rather than an if/elif chain, so it stays O(1) and doesn't
    need editing every time a tool is added. Called only from
    controller.py, never from the LLM client."""

    tool_fn = TOOL_REGISTRY.get(name)
    if tool_fn is None:
        return ToolResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}",
        )
    try:
        return tool_fn(arguments, user_id=user_id)
    except Exception as e:  # a broken tool must not crash the whole loop
        return ToolResult(tool_name=name, success=False, error=str(e))

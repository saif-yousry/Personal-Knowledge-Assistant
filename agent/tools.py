"""
Tool implementations for the agentic LLM.

Every function must:
  1. Accept a single `arguments: dict`.
  2. Return a `ToolResult` — never raise for expected failures.
  3. Be pure with respect to AgentState.
  4. Get registered in `tool_executer.py`'s TOOL_REGISTRY.
"""

from __future__ import annotations

from agent.tool_result import ToolResult
from initializer import store
from services.email_service import (
    _current_email_reply_to,
    email_service,
    EmailDeliveryError,
)

VALID_SOURCES = {"email", "slack", "discord", "telegram", "pdf"}


def search(arguments: dict) -> ToolResult:
    """Search the vector store, optionally filtered by source type."""
    query = arguments.get("query")
    if not query:
        return ToolResult(
            tool_name="search",
            success=False,
            error="Missing required argument: query",
        )

    source = arguments.get("source")
    if source and source not in VALID_SOURCES:
        return ToolResult(
            tool_name="search",
            success=False,
            error=f"Invalid source: '{source}'. Must be one of: {', '.join(sorted(VALID_SOURCES))}",
        )
    # Limit the number of results to a maximum of 10.
    # min() returns the minimum of two values. 
    k = min(arguments.get("num_results", 3), 5)
    results = store.similarity_search(query, k=k, source_type=source)

    if not results:
        return ToolResult(
            tool_name="search",
            success=True,
            data={"results": [], "message": "No matching documents found."},
        )

    return ToolResult(
        tool_name="search",
        success=True,
        data={"results": results},
    )


search.schema = {
    "description": "Search ingested content by semantic similarity. Set 'source' to filter by platform.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "source": {
                "type": "string",
                "enum": ["email", "slack", "discord", "telegram", "pdf"],
                "description": "Filter by source type. Omit to search all.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default 3).",
            },
        },
        "required": ["query"],
    },
}


def send_email_reply(arguments: dict) -> ToolResult:
    """Send a reply to an incoming email on behalf of the authenticated user."""
    # Resolve the recipient from trusted request context, not model-generated arguments.
    # (auto responder added part by saif)
    to = _current_email_reply_to.get()
    subject = arguments.get("subject", "").strip()
    body = arguments.get("body", "").strip()

    if not to:
        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error="No trusted sender address is available for this incoming email.",
        )
    if not body:
        return ToolResult(tool_name="send_email_reply", success=False,
                          error="Missing required argument: body")

    try:
        result = email_service.send_email(to=to, subject=subject, body=body)
        return ToolResult(
            tool_name="send_email_reply",
            success=True,
            data={"message": f"Email sent successfully to {to}.", "details": result},
        )
    except EmailDeliveryError as exc:
        return ToolResult(tool_name="send_email_reply", success=False, error=str(exc))
    except Exception as exc:
        return ToolResult(tool_name="send_email_reply", success=False,
                          error=f"Unexpected error while sending email: {exc}")


send_email_reply.schema = {
    "description": (
        "Send a reply to an incoming email using the authenticated user's linked Gmail account."
    ),
    "parameters": {
        # Only the reply content is model-controlled; identity and destination stay internal.
        # (auto responder added part by saif)
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Subject line of the email.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text body of the email.",
            },
        },
        "required": ["body"],
    },
}

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

VALID_SOURCES = {"email", "slack", "discord", "telegram", "pdf"}


def search(arguments: dict, user_id: int | None = None) -> ToolResult:
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

    # number of results to return, defaulting to 2 if not specified
    k = arguments.get("num_results", 2)
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
                "description": "Number of results (default 5).",
            },
        },
        "required": ["query"],
    },
}


# Reason: Added `send_email` tool function to enable the agentic LLM to compose and
# dispatch email responses to recipients. This is required so the agent can autonomously
# handle incoming emails, reply to inquiries, and communicate externally.
def send_email(arguments: dict, user_id: int | None = None) -> ToolResult:
    """Send an email using the configured email delivery service."""
    to = arguments.get("to")
    if not to:
        return ToolResult(
            tool_name="send_email",
            success=False,
            error="Missing required argument: 'to' (recipient email address).",
        )

    subject = arguments.get("subject")
    if not subject:
        return ToolResult(
            tool_name="send_email",
            success=False,
            error="Missing required argument: 'subject'.",
        )

    body = arguments.get("body")
    if body is None:
        return ToolResult(
            tool_name="send_email",
            success=False,
            error="Missing required argument: 'body'.",
        )

    if isinstance(to, list):
        to_str = ", ".join(to)
    else:
        to_str = str(to).strip()

    try:
        from services.email_service import email_service
        result_data = email_service.send_email(
            to=to_str,
            subject=str(subject),
            body=str(body),
            user_id=user_id,
        )
        return ToolResult(
            tool_name="send_email",
            success=True,
            data=result_data,
        )
    except Exception as exc:
        return ToolResult(
            tool_name="send_email",
            success=False,
            error=f"Failed to send email: {exc}",
        )


# Reason: Function-calling schema for `send_email` consumed by `llm_client._build_tools`.
# Defines the expected parameters (to, subject, body) so the LLM knows how to invoke the tool.
send_email.schema = {
    "description": "Send an email message to a specified recipient with a given subject and body.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address (e.g. 'user@example.com').",
            },
            "subject": {
                "type": "string",
                "description": "Subject line of the email.",
            },
            "body": {
                "type": "string",
                "description": "The plain text body/content of the email to send.",
            },
        },
        "required": ["to", "subject", "body"],
    },
}


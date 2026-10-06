"""
Agentic orchestrator loop.

Takes a conversation state, repeatedly calls the LLM, executes any
requested tools, and returns the final text answer. The controller
never talks to the provider SDK directly — that's llm_client.py's job.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agent.state import AgentState
from agent.llm_client import call_llm, LLMResponseError
from agent.tool_executer import execute_tool
from agent.system_prompt import SYSTEM_PROMPT
from agent.tool_result import ToolResult
from services.email_service import _current_email_reply_to, _current_user_id, email_service

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


def run(
    state: AgentState,
    system_prompt: Optional[str] = None,
    user_id: Optional[int] = None,
    allow_email_reply: bool = False,
    email_reply_to: Optional[str] = None,
) -> str:
    """
    Run the agent loop until the model produces a final answer.

    Parameters
    ----------
    state:
        Conversation history and current user message.
    system_prompt:
        Override the default SYSTEM_PROMPT.
    user_id:
        If provided, stored in a ContextVar so the email reply tool can
        authenticate without the LLM ever seeing the value.
    allow_email_reply:
        Expose and permit the email-reply tool only for incoming-email requests.
    email_reply_to:
        Trusted recipient for the current incoming email; never exposed as a tool argument.

    Returns the final text response to the user.
    """
    prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    # Store user_id in context so the email reply tool can authenticate securely.
    token = _current_user_id.set(user_id)
    # Pass trusted email context privately; never include it in the model's tool schema.
    # (auto responder added part by saif)
    reply_to_token = _current_email_reply_to.set(email_reply_to)
    email_service.clear_current_sent_email()
    # Require a successful knowledge-base lookup before allowing a reply or final answer.
    # (auto responder added part by saif)
    search_completed = False

    try:
        for round_num in range(MAX_TOOL_ROUNDS):
            try:
                message = call_llm(
                    state.messages,
                    prompt,
                    allow_email_reply=allow_email_reply,
                    force_search=not search_completed,
                )
            except LLMResponseError as e:
                logger.error("LLM call failed: %s", e)
                return f"Sorry, something went wrong: {e}"

            # No tool calls — model is done
            if not message.tool_calls:
                return message.content or "I have no response."

            # Append the assistant message (with tool_calls) to history
            # Only include fields Groq accepts — exclude SDK extras like "annotations"
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            state.messages.append(assistant_msg)

            # Execute each tool call
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                logger.info("Tool call: %s", name)

                if name == "send_email_reply" and not allow_email_reply:
                    # Regular chat cannot use the email reply capability.
                    # (auto responder added part by saif)
                    result = ToolResult(
                        tool_name=name,
                        success=False,
                        error="Email replies are only available while processing an incoming email.",
                    )
                elif name == "send_email_reply" and not search_completed:
                    # Enforce search-before-reply even if the model requests tools out of order.
                    # (auto responder added part by saif)
                    result = ToolResult(
                        tool_name=name,
                        success=False,
                        error="Search the email knowledge base before composing a reply.",
                    )
                else:
                    if name == "search" and allow_email_reply:
                        # Email replies must use email-history context only.
                        # (auto responder added part by saif)
                        args["source"] = "email"
                    result = execute_tool(name, args)
                    if name == "search" and result.success:
                        search_completed = True
                logger.info("Tool %s completed (success=%s).", name, result.success)

                state.add_tool_result(
                    tool_call_id=tool_call.id,
                    content=result.to_observation_text(),
                )

        return "I exceeded the maximum number of tool calls. Please try again."
    finally:
        # Always restore the ContextVar to avoid leaking user_id across requests.
        _current_email_reply_to.reset(reply_to_token)
        _current_user_id.reset(token)

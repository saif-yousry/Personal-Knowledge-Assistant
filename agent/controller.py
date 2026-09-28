"""
Agentic orchestrator loop.

Takes a conversation state, repeatedly calls the LLM, executes any
requested tools, and returns the final text answer. The controller
never talks to the provider SDK directly — that's llm_client.py's job.
"""

from __future__ import annotations

import json
import logging

from agent.state import AgentState
from agent.llm_client import call_llm, LLMResponseError
from agent.tool_executer import execute_tool
from agent.system_prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


# Reason: Added optional `system_prompt` argument so callers can pass specialized prompts
# (e.g. `EMAIL_AGENT_SYSTEM_PROMPT` for incoming email automation) while maintaining backward
# compatibility with existing chat routes by defaulting to `SYSTEM_PROMPT`.
def run(
    state: AgentState,
    system_prompt: str | None = None,
    *,
    user_id: int | None = None,
) -> str:
    """
    Run the agent loop until the model produces a final answer.
    Returns the final text response to the user.
    """
    active_prompt = system_prompt or SYSTEM_PROMPT
    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            message = call_llm(state.messages, active_prompt, user_id=user_id)
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
            logger.info("Tool call: %s(%s)", name, args)

            result = execute_tool(name, args, user_id=user_id)
            logger.info("Tool result: %s", result.to_observation_text())

            state.add_tool_result(
                tool_call_id=tool_call.id,
                content=result.to_observation_text(),
            )

    return "I exceeded the maximum number of tool calls. Please try again."

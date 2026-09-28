"""
Sole job: take the running message history, call the LLM provider, and
return the response. Nothing else in the codebase should talk to the
provider SDK directly — that keeps controller.py provider-agnostic, so
swapping OpenAI/Anthropic/local model later only touches this file.

Uses native tool-use (function calling) — the SDK returns structured
tool_calls objects directly, no manual JSON parsing needed.
"""

from typing import Any

from groq import Groq, GroqError
from pydantic import SecretStr

from config import settings
from agent.tool_executer import TOOL_REGISTRY
import logging

logging.basicConfig(level=logging.INFO)

_user_groq_api_keys: dict[int, SecretStr] = {}


def set_user_groq_api_key(user_id: int, api_key: str) -> None:
    _user_groq_api_keys[user_id] = SecretStr(api_key)


def clear_user_groq_api_key(user_id: int) -> None:
    _user_groq_api_keys.pop(user_id, None)


def get_user_groq_api_key_source(user_id: int) -> str | None:
    if user_id in _user_groq_api_keys:
        return "user"
    if settings.GROQ_API_KEY.get_secret_value():
        return "environment"
    return None


class LLMResponseError(Exception):
    """Raised when the provider's output can't be processed.
    Controller catches this and decides whether to retry, nudge
    the model, or bail out with an error to the user."""


def _get_client(user_id: int | None = None) -> Groq:
    user_api_key = _user_groq_api_keys.get(user_id) if user_id is not None else None
    api_key = (
        user_api_key.get_secret_value()
        if user_api_key is not None
        else settings.GROQ_API_KEY.get_secret_value()
    )
    if not api_key:
        raise LLMResponseError("Groq API key is not configured for this user.")
    return Groq(api_key=api_key)


def _build_tools() -> list[dict[str, Any]]:
    """Convert TOOL_REGISTRY into the OpenAI function-calling format.

    Each tool function should have a `schema` attribute (a dict with
    "description" and "parameters") for its definition. Tools without
    a schema are skipped.
    """
    tools = []
    for name, fn in TOOL_REGISTRY.items():
        schema = getattr(fn, "schema", None)
        if schema is None:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            },
        })
    return tools


def call_llm(
    messages: list[dict],
    system_prompt: str,
    user_id: int | None = None,
) -> dict:
    """Send the conversation to Groq and return the raw response message.

    Returns the assistant message dict from the API. The caller
    (controller) inspects `message.tool_calls` to decide whether
    to execute tools or treat `message.content` as the final answer.
    """
    client = _get_client(user_id)

    provider_messages = [
        {"role": "system", "content": system_prompt},
        *messages,
    ]

    tools = _build_tools()
    kwargs = {
        "model": settings.GROQ_MODEL,
        "messages": provider_messages,
        "temperature": 0,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    logger = logging.getLogger("uvicorn")

    try:
        completion = client.chat.completions.create(**kwargs)
        completion2 = client.chat.completions.create(**kwargs).model_dump_json()
        logging.info("Groq response: %s", completion2)
    except GroqError as e:
        raise LLMResponseError(f"Groq request failed: {e}") from e

    message = completion.choices[0].message
    if not message.content and not message.tool_calls:
        raise LLMResponseError("Groq returned an empty response.")

    return message

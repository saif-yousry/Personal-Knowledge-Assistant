"""
routes/chat.py

Endpoint for the agentic chat. Maintains per-session conversation
history in memory and delegates to the controller for LLM + tool execution.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends

from agent.llm_client import (
    clear_user_groq_api_key,
    get_user_groq_api_key_source,
    set_user_groq_api_key,
)
from services.app_auth_service import get_current_user

from agent.state import AgentState
from agent.controller import run
from schemas.chat import ChatPostRequest, ChatPostResponse, GroqApiKeyRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

# In-memory conversation sessions keyed by "{user_id}:{session_id}"
_sessions: dict[str, list[dict]] = {}


def _session_key(user_id: int, session_id: str) -> str:
    return f"{user_id}:{session_id}"


@router.get("/settings/groq-api-key")
def get_groq_api_key_status(user=Depends(get_current_user)):
    """Report whether this user has a Groq key configured without returning it."""
    source = get_user_groq_api_key_source(user.id)
    return {"configured": source is not None, "source": source}


@router.put("/settings/groq-api-key")
def save_groq_api_key(req: GroqApiKeyRequest, user=Depends(get_current_user)):
    """Keep the user's Groq API key in server memory for this process only."""
    api_key = req.api_key.strip()
    if not api_key:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Groq API key cannot be blank.",
        )
    set_user_groq_api_key(user.id, api_key)
    return {"configured": True, "source": "user"}


@router.delete("/settings/groq-api-key")
def delete_groq_api_key(user=Depends(get_current_user)):
    """Remove this user's in-memory Groq API key."""
    clear_user_groq_api_key(user.id)
    source = get_user_groq_api_key_source(user.id)
    return {"configured": source is not None, "source": source}


@router.post(
    "/chat",
    response_model=ChatPostResponse,
    summary="Send a message to the agent",
    description="Sends a user message to the agentic LLM and returns the response.",
)
def chat(req: ChatPostRequest, user=Depends(get_current_user)):
    """Send a message to the agent and return the response."""
    session_id = req.session_id or str(uuid.uuid4())
    key = _session_key(user.id, session_id)

    messages = _sessions.get(key, [])
    state = AgentState(user_input=req.message, messages=messages)

    reply = run(state, user_id=user.id)

    state.add_message(role="assistant", content=reply)
    _sessions[key] = state.messages

    return ChatPostResponse(reply=reply, session_id=session_id)


@router.post(
    "/chat/reset",
    summary="Reset chat session",
    description="Clears the conversation history for a session.",
)
def reset_chat(session_id: str, user=Depends(get_current_user)):
    """Clear the conversation history for a session."""
    key = _session_key(user.id, session_id)
    _sessions.pop(key, None)
    return {"status": "Session reset", "session_id": session_id}

"""
routes/chat.py

Endpoint for the agentic chat. Maintains per-session conversation
history in memory and delegates to the controller for LLM + tool execution.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends

from services.app_auth_service import get_current_user

from agent.state import AgentState
from agent.controller import run
from schemas.chat import ChatPostRequest, ChatPostResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

# In-memory conversation sessions keyed by "{user_id}:{session_id}"
_sessions: dict[str, list[dict]] = {}


def _session_key(user_id: int, session_id: str) -> str:
    return f"{user_id}:{session_id}"


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

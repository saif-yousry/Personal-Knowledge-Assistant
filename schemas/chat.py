"""
schemas/chat.py
Schemas for request and response because FastAPI requires them for validating and documenting the body of POST requests.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatPostRequest(BaseModel):
    """Incoming chat message with optional session ID."""
    message: str = Field(..., max_length=32000)
    session_id: str | None = Field(default=None, description="Session ID for continuing a conversation. Omit to start a new one.")


class ChatPostResponse(BaseModel):
    """Agent reply with the session ID."""
    reply: str
    session_id: str

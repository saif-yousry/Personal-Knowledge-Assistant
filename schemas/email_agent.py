"""
schemas/email_agent.py

Reason: Internal Pydantic schema for normalized incoming email data used by the
Gmail autoresponder service.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


# Reason: Internal input model that accepts structured email fields or raw MIME content.
class IncomingEmailRequest(BaseModel):
    """Incoming email data passed from Gmail sync to the email processing service."""

    sender: EmailStr = Field(..., description="Email address of the sender")
    subject: str = Field(default="", description="Subject line of the incoming email")
    body: str = Field(..., description="Plain text or HTML body of the email")
    recipients: Optional[List[EmailStr]] = Field(
        default=None, description="List of recipient email addresses (defaults to assistant/user address)"
    )
    raw_email: Optional[str] = Field(
        default=None, description="Optional raw RFC 822 / MIME email string"
    )

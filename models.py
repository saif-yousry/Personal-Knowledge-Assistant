"""
models/models.py

Pydantic data models for all content types entering the RAG pipeline.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ── Attachment ──────────────────────────────────────────────────────

class Attachment(BaseModel):
    """Represents a single email attachment."""

    filename: str = Field(..., min_length=1, description="Attachment file name")
    content_type: Optional[str] = Field(
        default=None, description="MIME type, e.g. application/pdf"
    )
    size: Optional[int] = Field(
        default=None, ge=0, description="Size of the attachment in bytes"
    )

    @field_validator("filename")
    @classmethod
    def filename_not_blank(cls, v: str) -> str:
        """Validate that the filename is not blank."""
        if not v.strip():
            raise ValueError("filename must not be blank")
        return v.strip()


# ── Email ───────────────────────────────────────────────────────────

class Email(BaseModel):
    """Represents a single validated/modeled email."""

    id: str = Field(..., min_length=1, description="Unique identifier of the email")
    subject: str = Field(default="", description="Email subject line")
    sender: EmailStr = Field(..., description="Sender's email address")
    recipients: List[EmailStr] = Field(
        default_factory=list, description="List of recipient email addresses"
    )
    body: str = Field(default="", description="Plain text or HTML body of the email")
    date: datetime = Field(..., description="Date/time the email was sent")
    attachments: List[Attachment] = Field(
        default_factory=list, description="List of attachments, if any"
    )
    gmail_thread_id: Optional[str] = Field(
        default=None, description="Gmail thread containing this message"
    )
    gmail_label_ids: List[str] = Field(
        default_factory=list, description="Gmail labels currently applied to this message"
    )
    gmail_internal_date_ms: Optional[int] = Field(
        default=None, description="Gmail's server-assigned message timestamp in milliseconds"
    )

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v: str) -> str:
        """Validate that the email id is not blank."""
        if not v.strip():
            raise ValueError("id must not be blank")
        return v.strip()

    model_config = ConfigDict(from_attributes=True)


# ── Document (PDF) ─────────────────────────────────────────────────

class Document(BaseModel):
    """Represents a single page extracted from a PDF."""

    source: str = Field(..., min_length=1, description="Source filename")
    page: int = Field(..., ge=1, description="Page number (1-based)")
    doc_type: str = Field(default="pdf", description="Document type")
    text: str = Field(..., min_length=1, description="Extracted text content")

    @field_validator("source")
    @classmethod
    def source_not_blank(cls, v: str) -> str:
        """Validate that the source filename is not blank."""
        if not v.strip():
            raise ValueError("source must not be blank")
        return v.strip()

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        """Validate that the text content is not blank."""
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


# ── BaseMessage (shared by Slack, Discord, Telegram) ───────────────

class BaseMessage(BaseModel):
    """Shared fields and validators for all platform message models."""

    id: str = Field(..., min_length=1, description="Unique message identifier")
    text: str = Field(..., min_length=1, description="Cleaned message text")
    timestamp: datetime = Field(..., description="When the message was sent")
    is_duplicate_of: Optional[str] = Field(default=None, description="Message ID this is a duplicate of")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be blank")
        return v.strip()

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v.strip()


# ── SlackMessage ────────────────────────────────────────────────────

class SlackMessage(BaseMessage):
    """Represents a single cleaned Slack message ready for the RAG pipeline."""

    channel_id: str = Field(..., min_length=1, description="Slack channel ID")
    channel_name: str = Field(default="", description="Human-readable channel name")
    user_id: str = Field(default="unknown", description="Slack user ID who sent the message")
    username: str = Field(default="unknown", description="Resolved display name")
    thread_ts: Optional[str] = Field(default=None, description="Parent thread timestamp, if a reply")


# ── DiscordMessage ──────────────────────────────────────────────────

class DiscordMessage(BaseMessage):
    """Represents a single cleaned Discord message ready for the RAG pipeline."""

    guild_id: str = Field(..., min_length=1, description="Discord guild (server) ID")
    channel_id: str = Field(..., min_length=1, description="Discord channel ID")
    channel_name: str = Field(default="", description="Human-readable channel name")
    author_id: str = Field(default="unknown", description="Discord user ID who sent the message")
    author_name: str = Field(default="unknown", description="Display name of the author")
    reply_to_message_id: Optional[str] = Field(default=None, description="Message ID this is a reply to")


# ── TelegramMessage ────────────────────────────────────────────────

class TelegramMessage(BaseMessage):
    """Represents a single cleaned Telegram message ready for the RAG pipeline."""

    chat_id: str = Field(..., min_length=1, description="Telegram chat ID")
    chat_title: str = Field(default="", description="Human-readable chat title")
    chat_type: str = Field(default="private", description="Chat type: private, group, supergroup, or channel")
    user_id: str = Field(default="unknown", description="Telegram user ID of the sender")
    username: str = Field(default="unknown", description="Display name of the sender")
    reply_to_message_id: Optional[str] = Field(default=None, description="Message ID this is a reply to")

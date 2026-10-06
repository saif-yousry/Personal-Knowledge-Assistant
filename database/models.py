"""
SQLAlchemy ORM models for the application database.
A SQLAlchemy engine is what holds the connections to the database.
There is one single engine object for all the code to connect to the same database.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    google_credentials: Mapped[GoogleCredentialRow | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    slack_credentials: Mapped[list[SlackCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    discord_credentials: Mapped[list[DiscordCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    telegram_credentials: Mapped[list[TelegramCredentialRow]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class CredentialRow(Base):
    """Abstract base for OAuth credential tables — shared columns only."""

    __abstract__ = True

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class GoogleCredentialRow(CredentialRow):
    __tablename__ = "google_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    google_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    token_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    history_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    backfill_cursor: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sync_label: Mapped[str] = mapped_column(Text, nullable=False, server_default="INBOX")
    autoresponder_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    user: Mapped[User] = relationship(back_populates="google_credentials")


class GmailReplyStateRow(Base):
    __tablename__ = "gmail_reply_states"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    message_id: Mapped[str] = mapped_column(Text, nullable=False)
    thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reply_message_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("user_id", "message_id", name="uq_gmail_reply_state_user_message"),
    )


class SlackCredentialRow(CredentialRow):
    """Stores Slack OAuth tokens linked to a workspace (team_id)."""

    __tablename__ = "slack_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    team_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    bot_user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    user: Mapped[User] = relationship(back_populates="slack_credentials")


class DiscordCredentialRow(CredentialRow):
    """Stores Discord OAuth tokens per user+guild connection."""

    __tablename__ = "discord_credentials"

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    discord_user_id: Mapped[str] = mapped_column(String(32), nullable=False)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False)
    token_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Bearer")
    scopes: Mapped[str] = mapped_column(Text, nullable=False, default="identify bot")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    granted_permissions: Mapped[str | None] = mapped_column(String(32), nullable=True)

    user: Mapped[User] = relationship(back_populates="discord_credentials")

    __table_args__ = (
        UniqueConstraint("discord_user_id", "guild_id", name="uq_discord_user_guild"),
    )


class TelegramCredentialRow(Base):
    """Stores Telegram session strings linked to a phone number."""

    __tablename__ = "telegram_credentials"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    phone_number: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    telegram_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    session_string: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    user: Mapped[User] = relationship(back_populates="telegram_credentials")


class CleanedMessageRow(Base):
    """Stores cleaned messages ready for RAG chunking and embedding."""

    __tablename__ = "cleaned_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False, default="slack")
    channel_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    is_boilerplate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_duplicate_of: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedup_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

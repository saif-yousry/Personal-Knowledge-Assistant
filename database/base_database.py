"""Abstract base class for all database implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import DiscordCredentialRow, GoogleCredentialRow, SlackCredentialRow, TelegramCredentialRow, User


class BaseDatabase(ABC):
    """Defines the contract for database access layers.

    Any concrete implementation (PostgreSQL, SQLite, etc.) must
    implement all methods below.
    """

    @abstractmethod
    def create_database(self) -> None:
        """Create the application database if it does not exist."""

    @abstractmethod
    def create_tables(self) -> None:
        """Create all application tables if they do not already exist."""

    @abstractmethod
    def find_user_by_id(self, session: Session, user_id: int) -> Optional[User]:
        """Return the user row for `user_id`, or None if not found."""

    @abstractmethod
    def user_exists(self, session: Session, email: str) -> bool:
        """Return True if a user with the given email exists."""

    @abstractmethod
    def find_user_by_email(self, session: Session, email: str) -> Optional[User]:
        """Return the user row for `email`, or None if not found."""

    @abstractmethod
    def create_user(self, session: Session, *, name: str, email: str, password_hash: str) -> User:
        """Insert a new user and return the created row."""

    @abstractmethod
    def update_user_profile(self, session: Session, user_id: int, *, name: str) -> User:
        """Update mutable profile fields for a user."""

    @abstractmethod
    def find_google_credentials(self, session: Session, user_id: int) -> Optional[GoogleCredentialRow]:
        """Return the Google credentials row for a user, or None."""

    @abstractmethod
    def find_user_by_google_id(self, session: Session, google_id: str) -> Optional[GoogleCredentialRow]:
        """Return the Google credentials row for `google_id`, or None."""

    @abstractmethod
    def find_all_google_credentials(self, session: Session) -> List[GoogleCredentialRow]:
        """Return all Google credential rows (one per linked user)."""

    @abstractmethod
    def update_google_access_token(self, session: Session, user_id: int, access_token: str, token_expiry: Optional[datetime]) -> None:
        """Persist a refreshed access token and its expiry for a user."""

    @abstractmethod
    def update_google_history_id(
        self,
        session: Session,
        user_id: int,
        history_id: Optional[str],
    ) -> None:
        """Update or clear the Gmail historyId cursor for a user."""

    @abstractmethod
    def update_google_backfill_cursor(self, session: Session, user_id: int, cursor: Optional[datetime]) -> None:
        """Update the backfill cursor for a user. Pass None to mark backfill complete."""

    @abstractmethod
    def update_google_sync_label(self, session: Session, user_id: int, label: str) -> None:
        """Update the Gmail label/folder to sync for a user."""

    @abstractmethod
    def delete_google_credentials(self, session: Session, user_id: int) -> bool:
        """Delete Google credentials for a user. Returns True if deleted."""

    @abstractmethod
    def save_google_credentials(
        self,
        session: Session,
        user_id: int,
        *,
        google_id: str,
        access_token: str,
        refresh_token: Optional[str],
        token_expiry: Optional[datetime],
    ) -> GoogleCredentialRow:
        """Insert or update Google credentials for a user."""

    # ------------------------------------------------------------------
    # Slack Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_slack_credentials(self, session: Session, team_id: str) -> Optional[SlackCredentialRow]:
        """Return the Slack credentials row for a team, or None."""

    @abstractmethod
    def save_slack_credentials(
        self,
        session: Session,
        user_id: int,
        team_id: str,
        *,
        access_token: str,
        refresh_token: Optional[str] = None,
        team_name: Optional[str] = None,
        bot_user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SlackCredentialRow:
        """Insert or update Slack credentials for a workspace."""

    @abstractmethod
    def delete_slack_credentials(self, session: Session, team_id: str) -> bool:
        """Delete Slack credentials for a workspace. Returns True if deleted."""

    # ------------------------------------------------------------------
    # Discord Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_discord_credentials(
        self, session: Session, discord_user_id: str, guild_id: str
    ) -> Optional[DiscordCredentialRow]:
        """Return the Discord credentials for a user+guild, or None."""

    @abstractmethod
    def find_discord_credentials_by_guild(self, session: Session, guild_id: str) -> Optional[DiscordCredentialRow]:
        """Return the first Discord credentials for a guild, or None."""

    @abstractmethod
    def save_discord_credentials(
        self,
        session: Session,
        *,
        user_id: int,
        discord_user_id: str,
        guild_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_type: str = "Bearer",
        scopes: str = "identify bot",
        expires_at: Optional[datetime] = None,
        granted_permissions: Optional[str] = None,
    ) -> DiscordCredentialRow:
        """Insert or update Discord credentials for a user+guild."""

    @abstractmethod
    def delete_discord_credentials(self, session: Session, discord_user_id: str, guild_id: str) -> bool:
        """Delete Discord credentials. Returns True if deleted."""

    @abstractmethod
    def list_discord_guild_ids(self, session: Session) -> List[str]:
        """Return all distinct connected guild IDs."""

    @abstractmethod
    def discord_guild_connection_exists(self, session: Session, guild_id: str) -> bool:
        """Return True if a connection exists for the guild."""

    @abstractmethod
    def discord_guild_owned_by_user(self, session: Session, user_id: int, guild_id: str) -> bool:
        """Return True if the app user owns a credential for this guild."""

    @abstractmethod
    def list_user_discord_guild_ids(self, session: Session, user_id: int) -> List[str]:
        """Return distinct guild IDs connected by the given app user."""

    # ------------------------------------------------------------------
    # Telegram Credentials
    # ------------------------------------------------------------------

    @abstractmethod
    def find_telegram_credentials(self, session: Session, phone_number: str) -> Optional[TelegramCredentialRow]:
        """Return the Telegram credentials for a phone number, or None."""

    @abstractmethod
    def save_telegram_credentials(
        self,
        session: Session,
        user_id: int,
        phone_number: str,
        *,
        telegram_user_id: Optional[int] = None,
        session_string: str,
    ) -> TelegramCredentialRow:
        """Insert or update Telegram credentials for a phone number."""

    @abstractmethod
    def delete_telegram_credentials(self, session: Session, phone_number: str) -> bool:
        """Delete Telegram credentials. Returns True if deleted."""

    # ------------------------------------------------------------------
    # Cleaned Messages
    # ------------------------------------------------------------------

    @abstractmethod
    def save_cleaned_messages(
        self,
        session: Session,
        messages: List[Dict[str, Any]],
        provider: str = "slack",
    ) -> int:
        """Save a batch of cleaned messages. Returns count saved."""

    @abstractmethod
    def get_stored_messages(
        self,
        session: Session,
        channel_id: Optional[str] = None,
        provider: str = "slack",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored cleaned messages."""

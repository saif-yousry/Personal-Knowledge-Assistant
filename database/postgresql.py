"""
PostgreSQL data-access layer using SQLAlchemy ORM.

Responsibilities:
  - Initial setup: create the application database and all ORM tables.
  - CRUD: insert, query, and update user rows (OAuth credentials, profile).

All methods receive an externally-managed ``Session`` instance.
Use ``get_session()`` as a FastAPI dependency for request-scoped sessions,
or ``session_context()`` for background tasks and services.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Generator, List, Optional

import psycopg2
from psycopg2 import sql
from sqlalchemy import and_, create_engine, func, or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from config import Settings
from .base_database import BaseDatabase
from .models import Base, CleanedMessageRow, DiscordCredentialRow, GmailReplyStateRow, GoogleCredentialRow, SlackCredentialRow, TelegramCredentialRow, User

logger = logging.getLogger(__name__)


class Database(BaseDatabase):
    """PostgreSQL implementation of the database access layer."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine = create_engine(
            settings.database_url,
            pool_size=settings.POSTGRES_POOL_SIZE,
            pool_pre_ping=True,
        )
        self._session_factory = sessionmaker(bind=self._engine)

    def get_session(self) -> Generator[Session, None, None]:
        """Yield a session for FastAPI ``Depends`` (one session per request)."""
        with self._session_factory() as session:
            yield session

    @contextmanager
    def session_context(self) -> Generator[Session, None, None]:
        """Context manager for code outside the request cycle (background tasks, services)."""
        with self._session_factory() as session:
            yield session

    def create_database(self) -> None:
        """Create the application database if it does not exist."""
        conn = psycopg2.connect(
            host=self._settings.POSTGRES_HOST,
            port=self._settings.POSTGRES_PORT,
            dbname="postgres",
            user=self._settings.POSTGRES_USER,
            password=self._settings.POSTGRES_PASSWORD.get_secret_value(),
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname=%s",
                    (self._settings.POSTGRES_DB,),
                )
                if not cur.fetchone():
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(
                            sql.Identifier(self._settings.POSTGRES_DB)
                        )
                    )
                    logger.info("Database %s created", self._settings.POSTGRES_DB)
                else:
                    logger.info("Database %s already exists", self._settings.POSTGRES_DB)
        finally:
            conn.close()

    def create_tables(self) -> None:
        """Create all ORM-defined tables if they do not already exist."""
        Base.metadata.create_all(self._engine)
        with self._engine.begin() as connection:
            connection.exec_driver_sql(
                "ALTER TABLE google_credentials "
                "ADD COLUMN IF NOT EXISTS autoresponder_enabled BOOLEAN NOT NULL DEFAULT FALSE"
            )

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def find_user_by_id(self, session: Session, user_id: int) -> Optional[User]:
        return session.get(User, user_id)

    def user_exists(self, session: Session, email: str) -> bool:
        stmt = select(User.id).where(User.email == email)
        return session.execute(stmt).first() is not None

    def find_user_by_email(self, session: Session, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email)
        return session.scalars(stmt).first()

    def create_user(self, session: Session, *, name: str, email: str, password_hash: str) -> User:
        user = User(name=name, email=email, password_hash=password_hash)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    def update_user_profile(self, session: Session, user_id: int, *, name: str) -> User:
        """
        Session.get(Model, primary_key) is SQLAlchemy's shortcut for fetching by primary key.                                                                                                                                         
        # These do the same thing:                                                                                                                                
        user = session.get(User, user_id)                                                                     
        user = session.execute(select(User).where(User.id == user_id)).scalars().first()                                                                                                                   
        
        session.get() is preferred for primary key lookups because:
        - Shorter and clearer                                                                                            
        - Checks the identity map first — if the object is already loaded in this session, it returns it instantly without hitting the database        
        - select() always hits the database
        
        The codebase uses both patterns appropriately:
        - session.get() — when looking up by primary key (find_user_by_id, update_user_profile)
        - select().where() — when querying by non-PK columns (find_user_by_email, find_slack_credentials, etc.)
        """

        user = session.get(User, user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        user.name = name
        session.commit()
        session.refresh(user)
        return user

    # ------------------------------------------------------------------
    # Google Credentials
    # ------------------------------------------------------------------

    def find_google_credentials(self, session: Session, user_id: int) -> Optional[GoogleCredentialRow]:
        stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)
        return session.scalars(stmt).first()

    def find_all_google_credentials(self, session: Session) -> list[GoogleCredentialRow]:
        rows = list(session.scalars(select(GoogleCredentialRow)).all())
        for row in rows:
            session.expunge(row)
        return rows

    def update_google_access_token(self, session: Session, user_id: int, access_token: str, token_expiry: Optional[datetime]) -> None:
        row = session.scalars(select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)).first()
        if row:
            row.access_token = access_token
            row.token_expiry = token_expiry
            session.commit()

    def update_google_history_id(
        self,
        session: Session,
        user_id: int,
        history_id: Optional[str],
    ) -> None:
        row = session.scalars(select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)).first()
        if row:
            row.history_id = history_id
            session.commit()

    def update_google_backfill_cursor(self, session: Session, user_id: int, cursor: Optional[datetime]) -> None:
        row = session.scalars(select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)).first()
        if row:
            row.backfill_cursor = cursor
            session.commit()

    def update_google_sync_label(self, session: Session, user_id: int, label: str) -> None:
        row = session.scalars(select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)).first()
        if row:
            row.sync_label = label
            session.commit()

    def set_google_autoresponder_enabled(
        self, session: Session, user_id: int, enabled: bool
    ) -> bool:
        row = session.scalars(
            select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)
        ).first()
        if row is None:
            return False
        row.autoresponder_enabled = enabled
        session.commit()
        return True

    def claim_gmail_message_for_autoreply(
        self, session: Session, user_id: int, message_id: str, thread_id: Optional[str]
    ) -> str:
        """Claim a message, recover expired work, or report its current processing state."""
        stale_before = datetime.now(timezone.utc) - timedelta(minutes=15)
        statement = (
            pg_insert(GmailReplyStateRow)
            .values(
                user_id=user_id,
                message_id=message_id,
                thread_id=thread_id,
                status="processing",
            )
            .on_conflict_do_update(
                constraint="uq_gmail_reply_state_user_message",
                set_={
                    "status": "processing",
                    "thread_id": thread_id,
                    "updated_at": func.now(),
                },
                where=or_(
                    GmailReplyStateRow.status.in_(("pending", "deferred", "failed")),
                    and_(
                        GmailReplyStateRow.status == "processing",
                        GmailReplyStateRow.updated_at < stale_before,
                    ),
                ),
            )
            .returning(GmailReplyStateRow.id)
        )
        claimed = session.execute(statement).scalar_one_or_none() is not None
        session.commit()
        if claimed:
            return "claimed"
        current_status = session.scalars(
            select(GmailReplyStateRow.status).where(
                GmailReplyStateRow.user_id == user_id,
                GmailReplyStateRow.message_id == message_id,
            )
        ).first()
        return "processing" if current_status == "processing" else "complete"

    def ensure_gmail_message_pending(
        self, session: Session, user_id: int, message_id: str, thread_id: Optional[str]
    ) -> None:
        """Persist an unanswered message without marking it as indexed."""
        statement = (
            pg_insert(GmailReplyStateRow)
            .values(
                user_id=user_id,
                message_id=message_id,
                thread_id=thread_id,
                status="pending",
            )
            .on_conflict_do_nothing(constraint="uq_gmail_reply_state_user_message")
        )
        session.execute(statement)
        session.commit()

    def find_gmail_reply_state(
        self, session: Session, user_id: int, message_id: str
    ) -> Optional[GmailReplyStateRow]:
        statement = select(GmailReplyStateRow).where(
            GmailReplyStateRow.user_id == user_id,
            GmailReplyStateRow.message_id == message_id,
        )
        return session.scalars(statement).first()

    def save_gmail_reply_state(
        self,
        session: Session,
        user_id: int,
        message_id: str,
        thread_id: Optional[str],
        *,
        state: str,
        reply_message_id: Optional[str] = None,
        indexed: bool = False,
    ) -> None:
        """Record the reply outcome and, when applicable, successful indexing."""
        indexed_at = datetime.now(timezone.utc) if indexed else None
        statement = pg_insert(GmailReplyStateRow).values(
            user_id=user_id,
            message_id=message_id,
            thread_id=thread_id,
            status=state,
            reply_message_id=reply_message_id,
            indexed_at=indexed_at,
        )
        statement = statement.on_conflict_do_update(
            constraint="uq_gmail_reply_state_user_message",
            set_={
                "thread_id": thread_id,
                "status": state,
                "reply_message_id": reply_message_id,
                "indexed_at": func.coalesce(
                    GmailReplyStateRow.indexed_at, statement.excluded.indexed_at
                ),
            },
        )
        session.execute(statement)
        session.commit()

    def delete_google_credentials(self, session: Session, user_id: int) -> bool:
        row = session.scalars(select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)).first()
        if row:
            session.delete(row)
            session.commit()
            return True
        return False

    def find_user_by_google_id(self, session: Session, google_id: str) -> Optional[GoogleCredentialRow]:
        stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.google_id == google_id)
        return session.scalars(stmt).first()

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
        # Check if this Google account is already linked to a different user
        stmt = select(GoogleCredentialRow).where(
            GoogleCredentialRow.google_id == google_id,
            GoogleCredentialRow.user_id != user_id,
        )
        if session.scalars(stmt).first():
            raise ValueError("This Google account is already linked to another user.")

        # Upsert: find existing row for user or create new
        stmt = select(GoogleCredentialRow).where(GoogleCredentialRow.user_id == user_id)
        cred = session.scalars(stmt).first()

        if cred:
            cred.google_id = google_id
            cred.access_token = access_token
            if refresh_token is not None:
                cred.refresh_token = refresh_token
            cred.token_expiry = token_expiry
        else:
            cred = GoogleCredentialRow(
                user_id=user_id,
                google_id=google_id,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=token_expiry,
            )
            session.add(cred)

        session.commit()
        session.refresh(cred)
        return cred

    # ------------------------------------------------------------------
    # Slack Credentials
    # ------------------------------------------------------------------

    def find_slack_credentials(self, session: Session, team_id: str) -> Optional[SlackCredentialRow]:
        stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
        return session.scalars(stmt).first()

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
        stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
        cred = session.scalars(stmt).first()

        meta_str = json.dumps(metadata) if metadata else None

        if cred:
            cred.user_id = user_id
            cred.access_token = access_token
            if refresh_token is not None:
                cred.refresh_token = refresh_token
            if team_name is not None:
                cred.team_name = team_name
            if bot_user_id is not None:
                cred.bot_user_id = bot_user_id
            if meta_str is not None:
                cred.metadata_json = meta_str
        else:
            cred = SlackCredentialRow(
                user_id=user_id,
                team_id=team_id,
                access_token=access_token,
                refresh_token=refresh_token,
                team_name=team_name,
                bot_user_id=bot_user_id,
                metadata_json=meta_str,
            )
            session.add(cred)

        session.commit()
        session.refresh(cred)
        return cred

    def delete_slack_credentials(self, session: Session, team_id: str) -> bool:
        stmt = select(SlackCredentialRow).where(SlackCredentialRow.team_id == team_id)
        cred = session.scalars(stmt).first()
        if cred:
            session.delete(cred)
            session.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Discord Credentials
    # ------------------------------------------------------------------

    def find_discord_credentials(
        self, session: Session, discord_user_id: str, guild_id: str
    ) -> Optional[DiscordCredentialRow]:
        stmt = select(DiscordCredentialRow).where(
            DiscordCredentialRow.discord_user_id == discord_user_id,
            DiscordCredentialRow.guild_id == guild_id,
        )
        return session.scalars(stmt).first()

    def find_discord_credentials_by_guild(self, session: Session, guild_id: str) -> Optional[DiscordCredentialRow]:
        stmt = select(DiscordCredentialRow).where(
            DiscordCredentialRow.guild_id == guild_id
        )
        return session.scalars(stmt).first()

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
        stmt = select(DiscordCredentialRow).where(
            DiscordCredentialRow.discord_user_id == discord_user_id,
            DiscordCredentialRow.guild_id == guild_id,
        )
        cred = session.scalars(stmt).first()

        if cred:
            cred.user_id = user_id
            cred.access_token = access_token
            if refresh_token is not None:
                cred.refresh_token = refresh_token
            cred.token_type = token_type
            cred.scopes = scopes
            if expires_at is not None:
                cred.expires_at = expires_at
            cred.granted_permissions = granted_permissions
        else:
            cred = DiscordCredentialRow(
                user_id=user_id,
                discord_user_id=discord_user_id,
                guild_id=guild_id,
                access_token=access_token,
                refresh_token=refresh_token,
                token_type=token_type,
                scopes=scopes,
                expires_at=expires_at,
                granted_permissions=granted_permissions,
            )
            session.add(cred)

        session.commit()
        session.refresh(cred)
        return cred

    def delete_discord_credentials(self, session: Session, discord_user_id: str, guild_id: str) -> bool:
        stmt = select(DiscordCredentialRow).where(
            DiscordCredentialRow.discord_user_id == discord_user_id,
            DiscordCredentialRow.guild_id == guild_id,
        )
        cred = session.scalars(stmt).first()
        if cred:
            session.delete(cred)
            session.commit()
            return True
        return False

    def list_discord_guild_ids(self, session: Session) -> List[str]:
        return list(session.scalars(select(DiscordCredentialRow.guild_id).distinct()))

    def discord_guild_connection_exists(self, session: Session, guild_id: str) -> bool:
        stmt = select(DiscordCredentialRow.id).where(
            DiscordCredentialRow.guild_id == guild_id
        ).limit(1)
        return session.execute(stmt).first() is not None

    def discord_guild_owned_by_user(self, session: Session, user_id: int, guild_id: str) -> bool:
        stmt = select(DiscordCredentialRow.id).where(
            DiscordCredentialRow.user_id == user_id,
            DiscordCredentialRow.guild_id == guild_id,
        ).limit(1)
        return session.execute(stmt).first() is not None

    def list_user_discord_guild_ids(self, session: Session, user_id: int) -> List[str]:
        stmt = select(DiscordCredentialRow.guild_id).where(
            DiscordCredentialRow.user_id == user_id
        ).distinct()
        return list(session.scalars(stmt))

    # ------------------------------------------------------------------
    # Telegram Credentials
    # ------------------------------------------------------------------

    def find_telegram_credentials(self, session: Session, phone_number: str) -> Optional[TelegramCredentialRow]:
        stmt = select(TelegramCredentialRow).where(
            TelegramCredentialRow.phone_number == phone_number
        )
        return session.scalars(stmt).first()

    def save_telegram_credentials(
        self,
        session: Session,
        user_id: int,
        phone_number: str,
        *,
        telegram_user_id: Optional[int] = None,
        session_string: str,
    ) -> TelegramCredentialRow:
        stmt = select(TelegramCredentialRow).where(
            TelegramCredentialRow.phone_number == phone_number
        )
        cred = session.scalars(stmt).first()

        if cred:
            cred.user_id = user_id
            cred.session_string = session_string
            if telegram_user_id is not None:
                cred.telegram_user_id = telegram_user_id
        else:
            cred = TelegramCredentialRow(
                user_id=user_id,
                phone_number=phone_number,
                telegram_user_id=telegram_user_id,
                session_string=session_string,
            )
            session.add(cred)

        session.commit()
        session.refresh(cred)
        return cred

    def delete_telegram_credentials(self, session: Session, phone_number: str) -> bool:
        stmt = select(TelegramCredentialRow).where(
            TelegramCredentialRow.phone_number == phone_number
        )
        cred = session.scalars(stmt).first()
        if cred:
            session.delete(cred)
            session.commit()
            return True
        return False

    # ------------------------------------------------------------------
    # Cleaned Messages
    # ------------------------------------------------------------------

    def save_cleaned_messages(
        self,
        session: Session,
        messages: List[Dict[str, Any]],
        provider: str = "slack",
    ) -> int:
        saved_count = 0
        for msg in messages:
            meta = msg.get("metadata") or {}
            record = CleanedMessageRow(
                provider=provider,
                channel_id=msg.get("channel_id") or meta.get("channel"),
                message_id=msg.get("message_id") or meta.get("ts"),
                user_id=msg.get("user_id") or meta.get("user"),
                raw_text=msg.get("raw_text", ""),
                cleaned_text=msg.get("cleaned_text", ""),
                is_boilerplate=msg.get("is_boilerplate", False),
                is_duplicate_of=msg.get("is_duplicate_of"),
                dedup_hash=msg.get("dedup_hash"),
                metadata_json=json.dumps(meta) if meta else None,
            )
            session.add(record)
            saved_count += 1
        session.commit()
        return saved_count

    def get_stored_messages(
        self,
        session: Session,
        channel_id: Optional[str] = None,
        provider: str = "slack",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        stmt = select(CleanedMessageRow).where(CleanedMessageRow.provider == provider)
        if channel_id:
            stmt = stmt.where(CleanedMessageRow.channel_id == channel_id)
        stmt = stmt.order_by(CleanedMessageRow.id.desc()).limit(limit)
        records = session.scalars(stmt).all()
        results = []
        for r in records:
            meta = {}
            if r.metadata_json:
                try:
                    meta = json.loads(r.metadata_json)
                except Exception:
                    pass
            results.append({
                "id": r.id,
                "provider": r.provider,
                "channel_id": r.channel_id,
                "message_id": r.message_id,
                "user_id": r.user_id,
                "raw_text": r.raw_text,
                "cleaned_text": r.cleaned_text,
                "is_boilerplate": r.is_boilerplate,
                "is_duplicate_of": r.is_duplicate_of,
                "dedup_hash": r.dedup_hash,
                "metadata": meta,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            })
        return results

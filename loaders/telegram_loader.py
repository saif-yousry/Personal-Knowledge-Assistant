"""
telegram_loader.py

Fetches messages from Telegram chats via Telethon using saved session strings.

Data flow:
    Telegram MTProto  --(TelegramLoader)-->  List[Dict]   (raw message dicts)

This module only fetches and structures raw messages. It performs no
cleaning, chunking, embedding, storage, or retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession
from telethon.tl.types import Channel, Chat, Message, User

logger = logging.getLogger(__name__)


def _display_name(entity) -> str:
    """Best-effort human-readable name for a user/chat/channel entity."""
    if entity is None:
        return "Unknown"
    first = getattr(entity, "first_name", None)
    last = getattr(entity, "last_name", None)
    if first or last:
        return " ".join(p for p in (first, last) if p)
    title = getattr(entity, "title", None)
    if title:
        return title
    username = getattr(entity, "username", None)
    if username:
        return f"@{username}"
    return "Unknown"


def _chat_type(entity) -> str:
    """Determine the chat type string from a Telethon entity."""
    if isinstance(entity, User):
        return "private"
    if isinstance(entity, Channel):
        return "channel" if not entity.megagroup else "supergroup"
    if isinstance(entity, Chat):
        return "group"
    return "unknown"


class TelegramLoader:
    """Loads messages from Telegram chats using a saved session string."""

    def __init__(self, session_string: str, api_id: int, api_hash: str):
        self._session_string = session_string
        self._api_id = api_id
        self._api_hash = api_hash
        self._client: Optional[TelegramClient] = None

    async def connect(self) -> None:
        """Connect and verify authorization."""
        self._client = TelegramClient(
            StringSession(self._session_string), self._api_id, self._api_hash
        )
        await self._client.connect()
        if not await self._client.is_user_authorized():
            await self._client.disconnect()
            self._client = None
            raise RuntimeError("Telegram session is no longer valid; re-authenticate.")

    async def disconnect(self) -> None:
        if self._client and self._client.is_connected():
            await self._client.disconnect()

    async def __aenter__(self) -> TelegramLoader:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.disconnect()

    def _ensure_connected(self) -> TelegramClient:
        if self._client is None or not self._client.is_connected():
            raise RuntimeError("TelegramLoader is not connected. Call connect() first.")
        return self._client

    async def list_dialogs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Lists the most recently active chats."""
        client = self._ensure_connected()
        dialogs = []
        async for dialog in client.iter_dialogs(limit=limit):
            entity = dialog.entity
            dialogs.append({
                "id": dialog.id,
                "name": dialog.name or _display_name(entity),
                "type": _chat_type(entity),
                "unread_count": dialog.unread_count,
            })
        return dialogs

    async def _iter_with_retry(self, entity, **kwargs):
        """Wraps client.iter_messages with flood-wait retry."""
        client = self._ensure_connected()
        while True:
            try:
                async for msg in client.iter_messages(entity, **kwargs):
                    yield msg
                return
            except FloodWaitError as e:
                logger.warning("Flood wait %ds, sleeping...", e.seconds)
                await asyncio.sleep(e.seconds + 1)

    async def fetch_chat_messages(
        self,
        chat: Union[str, int],
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch up to `limit` text messages from a chat."""
        client = self._ensure_connected()
        entity = await client.get_entity(chat)
        chat_title = _display_name(entity)
        ct = _chat_type(entity)

        messages: List[Dict[str, Any]] = []
        async for msg in self._iter_with_retry(entity, limit=limit):
            if not isinstance(msg, Message) or not msg.text:
                continue

            sender_name = None
            try:
                sender = await msg.get_sender()
                sender_name = _display_name(sender)
            except Exception:
                pass

            messages.append({
                "message_id": msg.id,
                "chat_id": entity.id,
                "chat_title": chat_title,
                "chat_type": ct,
                "sender_id": msg.sender_id,
                "sender_name": sender_name,
                "text": msg.text,
                "date": msg.date,
                "reply_to_message_id": msg.reply_to_msg_id,
                "is_outgoing": bool(msg.out),
                "media_type": type(msg.media).__name__ if msg.media else None,
            })
            logger.info("Fetched message %d from chat %s", msg.id, chat_title)
        return messages

    async def fetch_all_messages(
        self,
        limit_per_chat: int = 50,
        dialog_limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch messages across all recent dialogs."""
        dialogs = await self.list_dialogs(limit=dialog_limit)
        all_messages: List[Dict[str, Any]] = []
        for dialog in dialogs:
            try:
                msgs = await self.fetch_chat_messages(
                    chat=dialog["id"], limit=limit_per_chat
                )
                all_messages.extend(msgs)
            except Exception as exc:
                logger.warning("Failed to fetch from %s: %s", dialog["name"], exc)
        return all_messages


def get_telegram_loader(phone_number: str) -> TelegramLoader:
    """Factory: build a TelegramLoader from a stored session."""
    from config import settings
    from services.telegram_auth_service import get_session_string

    api_hash = settings.TELEGRAM_API_HASH.get_secret_value()
    if not settings.TELEGRAM_API_ID or not api_hash:
        raise RuntimeError("Telegram API credentials not configured.")

    session_string = get_session_string(phone_number)
    return TelegramLoader(
        session_string=session_string,
        api_id=settings.TELEGRAM_API_ID,
        api_hash=api_hash,
    )

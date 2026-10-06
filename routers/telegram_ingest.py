"""
routers/telegram_ingest.py

Endpoints to trigger Telegram message ingestion into the RAG pipeline.

Flow:
  1. Fetch raw messages from Telegram via TelegramLoader.
  2. Build TelegramMessage models, clean, and deduplicate.
  3. Run through the RAG pipeline (chunk, embed, store).
  4. Optionally persist cleaned messages to PostgreSQL.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from services.app_auth_service import get_current_user

from initializer import db, embedder, store, dispatcher, chunker
from loaders.telegram_loader import TelegramLoader, get_telegram_loader
from models import TelegramMessage
from rag.pipeline import Pipeline
from rag.preprocessing.telegram_cleaner import TelegramCleaner, deduplicate_messages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["telegram-ingest"])

# In-memory ingestion progress tracker keyed by phone_number
_telegram_ingest_status: dict[str, dict] = {}


def _verify_phone_ownership(session: Session, user_id: int, phone_number: str) -> None:
    """Raise 403 if the authenticated user does not own this Telegram connection."""
    cred = db.find_telegram_credentials(session, phone_number)
    if not cred or cred.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this Telegram account.")


def _run_telegram_ingest(
    phone_number: str,
    chat_id: Optional[str],
    limit_per_chat: int,
    dialog_limit: int,
    save_to_db: bool,
) -> None:
    """Background task: fetch, clean, deduplicate, and ingest Telegram messages."""
    status_key = phone_number
    _telegram_ingest_status[status_key] = {"state": "fetching", "fetched": 0, "chunks_stored": 0}

    try:
        loader = get_telegram_loader(phone_number)

        # Run async loader in a new event loop (background tasks are sync)
        loop = asyncio.new_event_loop()
        try:
            if chat_id:
                raw_messages = loop.run_until_complete(
                    _fetch_chat(loader, chat_id, limit_per_chat)
                )
            else:
                raw_messages = loop.run_until_complete(
                    _fetch_all(loader, limit_per_chat, dialog_limit)
                )
        finally:
            loop.close()

        _telegram_ingest_status[status_key]["fetched"] = len(raw_messages)
        _telegram_ingest_status[status_key]["state"] = "cleaning"

        # Build TelegramMessage models and clean
        cleaner = TelegramCleaner()
        telegram_messages = []

        for m in raw_messages:
            text = m.get("text", "")
            if not text or not text.strip():
                continue

            try:
                msg = TelegramMessage(
                    id=f"{m['chat_id']}:{m['message_id']}",
                    chat_id=str(m["chat_id"]),
                    chat_title=m.get("chat_title", ""),
                    chat_type=m.get("chat_type", "private"),
                    user_id=str(m.get("sender_id") or "unknown"),
                    username=m.get("sender_name") or "unknown",
                    text=text,
                    timestamp=m.get("date") or datetime.now(timezone.utc),
                    reply_to_message_id=str(m["reply_to_message_id"]) if m.get("reply_to_message_id") else None,
                    metadata={
                        "is_outgoing": m.get("is_outgoing", False),
                        "media_type": m.get("media_type"),
                    },
                )
                cleaned = cleaner.clean(msg)
                if cleaned.text:
                    telegram_messages.append(cleaned)
            except Exception as exc:
                logger.warning("Skipping message %s: %s", m.get("message_id"), exc)

        # Deduplicate
        telegram_messages = deduplicate_messages(telegram_messages)
        unique_messages = [m for m in telegram_messages if m.is_duplicate_of is None]

        _telegram_ingest_status[status_key]["state"] = "processing"

        # Save cleaned messages to DB
        if save_to_db and telegram_messages:
            try:
                records = [
                    {
                        "channel_id": m.chat_id,
                        "message_id": m.id,
                        "user_id": m.user_id,
                        "raw_text": m.text,
                        "cleaned_text": m.text,
                        "is_boilerplate": not bool(m.text),
                        "is_duplicate_of": m.is_duplicate_of,
                        "metadata": m.metadata,
                    }
                    for m in telegram_messages
                ]
                with db.session_context() as bg_session:
                    db.save_cleaned_messages(bg_session, records, provider="telegram")
            except Exception as exc:
                logger.warning("Failed to persist cleaned messages to DB: %s", exc)

        # Run RAG pipeline
        pipeline = Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
        count = pipeline.run(unique_messages)

        _telegram_ingest_status[status_key]["chunks_stored"] = count
        _telegram_ingest_status[status_key]["state"] = "done"
        logger.info(
            "Telegram ingestion complete: %d messages fetched, %d unique, %d chunks stored.",
            len(raw_messages), len(unique_messages), count,
        )
    except Exception as exc:
        _telegram_ingest_status[status_key]["state"] = "error"
        _telegram_ingest_status[status_key]["error"] = str(exc)
        logger.exception("Telegram ingestion failed: %s", exc)


async def _fetch_chat(loader: TelegramLoader, chat_id: str, limit: int):
    """Fetch messages from a single chat."""
    async with loader:
        chat = int(chat_id) if chat_id.lstrip("-").isdigit() else chat_id
        return await loader.fetch_chat_messages(chat=chat, limit=limit)


async def _fetch_all(loader: TelegramLoader, limit_per_chat: int, dialog_limit: int):
    """Fetch messages across all recent dialogs."""
    async with loader:
        return await loader.fetch_all_messages(
            limit_per_chat=limit_per_chat, dialog_limit=dialog_limit
        )


@router.post(
    "/ingest/telegram",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Telegram message ingestion",
)
def ingest_telegram(
    background_tasks: BackgroundTasks,
    phone_number: str = Query(..., description="Phone number of the authenticated Telegram account"),
    chat_id: Optional[str] = Query(None, description="Chat ID (leave empty to sync all recent chats)"),
    limit_per_chat: int = Query(default=50, ge=1, le=500, description="Max messages per chat"),
    dialog_limit: int = Query(default=20, ge=1, le=100, description="Max number of chats to scan"),
    save_to_db: bool = Query(default=True, description="Persist cleaned messages to PostgreSQL"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Fetch Telegram messages, clean, deduplicate, and run the RAG pipeline in the background."""
    _verify_phone_ownership(session, user.id, phone_number)
    background_tasks.add_task(
        _run_telegram_ingest, phone_number, chat_id, limit_per_chat, dialog_limit, save_to_db
    )
    return {
        "status": "Telegram ingestion started",
        "phone_number": phone_number,
        "chat_id": chat_id or "all",
        "limit_per_chat": limit_per_chat,
    }


@router.get(
    "/ingest/telegram/status",
    summary="Check Telegram ingestion progress",
)
def telegram_ingest_status(
    phone_number: str = Query(..., description="Phone number"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Return the current Telegram ingestion progress."""
    _verify_phone_ownership(session, user.id, phone_number)
    status = _telegram_ingest_status.get(phone_number)
    if not status:
        return {"state": "idle"}
    return status


@router.get(
    "/telegram/chats",
    summary="List Telegram chats",
)
async def list_telegram_chats(
    phone_number: str = Query(..., description="Phone number of the authenticated Telegram account"),
    limit: int = Query(default=20, ge=1, le=100, description="Max dialogs to return"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Lists the most recently active Telegram chats."""
    _verify_phone_ownership(session, user.id, phone_number)
    try:
        loader = get_telegram_loader(phone_number)
        async with loader:
            dialogs = await loader.list_dialogs(limit=limit)
        return {"phone_number": phone_number, "chats_count": len(dialogs), "chats": dialogs}
    except Exception as exc:
        logger.error("Failed to list Telegram chats: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

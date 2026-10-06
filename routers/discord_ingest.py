"""
routers/discord_ingest.py

Endpoints to trigger Discord message ingestion into the RAG pipeline.

Flow:
  1. Fetch raw messages from Discord via DiscordLoader.
  2. Clean via DiscordCleaner.
  3. Build DiscordMessage models.
  4. Run through the RAG pipeline (chunk, embed, store).
  5. Optionally persist cleaned messages to PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from services.app_auth_service import get_current_user

from initializer import db, embedder, store, dispatcher, chunker
from loaders.discord_loader import (
    discover_guild_channels,
    list_connected_guilds,
    load_guild_messages,
    DiscordIntegrationError,
)
from models import DiscordMessage
from rag.pipeline import Pipeline
from rag.preprocessing.discord_cleaner import DiscordCleaner

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["discord-ingest"])

# In-memory ingestion progress tracker keyed by guild_id
_discord_ingest_status: dict[str, dict] = {}


def _verify_guild_ownership(session: Session, user_id: int, guild_id: str) -> None:
    """Raise 403 if the authenticated user does not own this guild connection."""
    if not db.discord_guild_owned_by_user(session, user_id, guild_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this Discord guild.")


def _run_discord_ingest(
    guild_id: str,
    max_per_channel: int,
    save_to_db: bool,
) -> None:
    """Background task: fetch, clean, and ingest Discord messages."""
    _discord_ingest_status[guild_id] = {"state": "fetching", "fetched": 0, "chunks_stored": 0}

    try:
        result = load_guild_messages(guild_id, max_messages_per_channel=max_per_channel)

        _discord_ingest_status[guild_id]["fetched"] = len(result.messages)
        _discord_ingest_status[guild_id]["state"] = "cleaning"

        cleaner = DiscordCleaner()

        discord_messages = []
        for msg in result.messages:
            if not msg.content or not msg.content.strip():
                continue

            try:
                dm = DiscordMessage(
                    id=msg.message_id,
                    guild_id=msg.guild_id,
                    channel_id=msg.channel_id,
                    channel_name=msg.channel_name or "",
                    author_id=msg.author_id,
                    author_name=msg.author_name or "unknown",
                    text=msg.content,
                    timestamp=msg.timestamp,
                    reply_to_message_id=msg.reply_to_message_id,
                    metadata={
                        "mentions": [m.model_dump() for m in msg.mentions],
                        "attachments": [a.model_dump() for a in msg.attachments],
                        "stickers": [s.model_dump() for s in msg.stickers],
                    },
                )
                cleaned = cleaner.clean(dm)
                if cleaned.text:
                    discord_messages.append(cleaned)
            except Exception as exc:
                logger.warning("Skipping Discord message %s: %s", msg.message_id, exc)

        _discord_ingest_status[guild_id]["state"] = "processing"

        # Save cleaned messages to DB
        if save_to_db and discord_messages:
            try:
                records = [
                    {
                        "channel_id": m.channel_id,
                        "message_id": m.id,
                        "user_id": m.author_id,
                        "raw_text": m.metadata.get("original_text", m.text),
                        "cleaned_text": m.text,
                        "is_boilerplate": not bool(m.text),
                        "is_duplicate_of": m.is_duplicate_of,
                        "metadata": m.metadata,
                    }
                    for m in discord_messages
                ]
                with db.session_context() as bg_session:
                    db.save_cleaned_messages(bg_session, records, provider="discord")
            except Exception as exc:
                logger.warning("Failed to persist cleaned Discord messages to DB: %s", exc)

        # Filter out duplicates before pipeline
        unique_messages = [m for m in discord_messages if m.is_duplicate_of is None]

        # Run RAG pipeline
        pipeline = Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
        count = pipeline.run(unique_messages)

        _discord_ingest_status[guild_id]["chunks_stored"] = count
        _discord_ingest_status[guild_id]["state"] = "done"
        logger.info(
            "Discord ingestion complete for guild %s: %d messages fetched, %d unique, %d chunks stored.",
            guild_id, len(result.messages), len(unique_messages), count,
        )
    except Exception as exc:
        _discord_ingest_status[guild_id]["state"] = "error"
        _discord_ingest_status[guild_id]["error"] = str(exc)
        logger.exception("Discord ingestion failed for guild %s: %s", guild_id, exc)


@router.post(
    "/ingest/discord",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Discord message ingestion",
)
def ingest_discord(
    background_tasks: BackgroundTasks,
    guild_id: str = Query(..., description="Discord guild (server) ID"),
    max_per_channel: int = Query(default=50, ge=1, le=500, description="Max messages per channel"),
    save_to_db: bool = Query(default=True, description="Persist cleaned messages to PostgreSQL"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Fetch Discord messages, clean, and run the RAG pipeline in the background."""
    _verify_guild_ownership(session, user.id, guild_id)
    background_tasks.add_task(_run_discord_ingest, guild_id, max_per_channel, save_to_db)
    return {
        "status": "Discord ingestion started",
        "guild_id": guild_id,
        "max_per_channel": max_per_channel,
    }


@router.get(
    "/ingest/discord/status",
    summary="Check Discord ingestion progress",
)
def discord_ingest_status(
    guild_id: str = Query(..., description="Discord guild ID"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Return the current Discord ingestion progress."""
    _verify_guild_ownership(session, user.id, guild_id)
    status = _discord_ingest_status.get(guild_id)
    if not status:
        return {"state": "idle"}
    return status


@router.get(
    "/discord/guilds",
    summary="List connected Discord guilds",
)
def list_guilds(
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Lists guilds the bot is connected to for the authenticated user."""
    try:
        user_guild_ids = set(db.list_user_discord_guild_ids(session, user.id))
        guilds = [g for g in list_connected_guilds() if g.guild_id in user_guild_ids]
        return {"guilds_count": len(guilds), "guilds": [g.model_dump() for g in guilds]}
    except DiscordIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to list guilds: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get(
    "/discord/channels",
    summary="List channels in a Discord guild",
)
def list_channels(
    guild_id: str = Query(..., description="Discord guild ID"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Lists text channels the bot can see in a guild."""
    _verify_guild_ownership(session, user.id, guild_id)
    try:
        channels = discover_guild_channels(guild_id)
        return {
            "guild_id": guild_id,
            "channels_count": len(channels),
            "channels": [c.model_dump() for c in channels],
        }
    except DiscordIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to list channels: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

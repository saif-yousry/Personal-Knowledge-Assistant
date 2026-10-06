"""
routers/slack_ingest.py

Endpoints to trigger Slack message ingestion into the RAG pipeline.

Flow:
  1. Fetch raw messages from Slack via SlackLoader.
  2. Clean and deduplicate via SlackCleaner.
  3. Build SlackMessage models.
  4. Run through the RAG pipeline (chunk, embed, store).
  5. Optionally persist cleaned messages to PostgreSQL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from services.app_auth_service import get_current_user

from initializer import db, embedder, store, dispatcher, chunker
from loaders.slack_loader import get_slack_loader_for_team
from models import SlackMessage
from rag.pipeline import Pipeline
from rag.preprocessing.slack_cleaner import SlackCleaner, SlackCleaningConfig, deduplicate_messages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["slack-ingest"])

# In-memory ingestion progress tracker keyed by team_id
_slack_ingest_status: dict[str, dict] = {}


def _verify_team_ownership(session: Session, user_id: int, team_id: Optional[str]) -> None:
    """Raise 403 if team_id is given and the authenticated user does not own it."""
    if not team_id:
        return
    cred = db.find_slack_credentials(session, team_id)
    if not cred or cred.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this Slack workspace.")


def _run_slack_ingest(
    team_id: Optional[str],
    channel_id: Optional[str],
    limit_per_channel: int,
    save_to_db: bool,
) -> None:
    """Background task: fetch, clean, deduplicate, and ingest Slack messages."""
    status_key = team_id or "default"
    _slack_ingest_status[status_key] = {"state": "fetching", "fetched": 0, "chunks_stored": 0}

    try:
        loader = get_slack_loader_for_team(team_id=team_id)

        # Discover channels
        try:
            channels = loader.list_user_channels()
        except Exception as exc:
            logger.warning("Failed to list channels: %s", exc)
            channels = []

        channel_map = {ch["id"]: ch["name"] for ch in channels}
        target_channels = [channel_id] if channel_id else [ch["id"] for ch in channels]

        if channel_id and channel_id not in channel_map:
            channel_map[channel_id] = channel_id

        # Fetch raw messages
        all_raw_messages = []
        for ch_id in target_channels:
            try:
                msgs = loader.fetch_channel_messages(channel_id=ch_id, limit=limit_per_channel)
                for m in msgs:
                    if "channel" not in m:
                        m["channel"] = ch_id
                all_raw_messages.extend(msgs)
            except Exception as exc:
                logger.error("Error fetching channel %s: %s", ch_id, exc)

        _slack_ingest_status[status_key]["fetched"] = len(all_raw_messages)
        _slack_ingest_status[status_key]["state"] = "cleaning"

        # Clean: build SlackMessage models from raw dicts, then run SlackCleaner
        cleaner = SlackCleaner(
            config=SlackCleaningConfig(),
            channel_id_map=channel_map,
        )

        slack_messages = []
        for m in all_raw_messages:
            text = m.get("text", "")
            if not text or not text.strip():
                continue

            ts = m.get("ts", "")
            try:
                msg = SlackMessage(
                    id=m.get("client_msg_id") or ts,
                    channel_id=m.get("channel", ""),
                    channel_name=channel_map.get(m.get("channel", ""), ""),
                    user_id=m.get("user", "unknown"),
                    username=m.get("user", "unknown"),
                    text=text,
                    timestamp=datetime.fromtimestamp(float(ts), tz=timezone.utc) if ts else datetime.now(timezone.utc),
                    thread_ts=m.get("thread_ts"),
                    metadata={
                        "subtype": m.get("subtype"),
                        "ts": ts,
                    },
                )
                cleaned = cleaner.clean(msg)
                if cleaned.text:
                    slack_messages.append(cleaned)
            except Exception as exc:
                logger.warning("Skipping message ts=%s: %s", ts, exc)

        # Deduplicate
        slack_messages = deduplicate_messages(slack_messages)

        # Filter out duplicates before pipeline
        unique_messages = [m for m in slack_messages if m.is_duplicate_of is None]

        _slack_ingest_status[status_key]["state"] = "processing"

        # Save cleaned messages to DB
        if save_to_db and slack_messages:
            try:
                records = [
                    {
                        "channel_id": m.channel_id,
                        "message_id": m.id,
                        "user_id": m.user_id,
                        "raw_text": m.metadata.get("original_text", m.text),
                        "cleaned_text": m.text,
                        "is_boilerplate": not bool(m.text),
                        "is_duplicate_of": m.is_duplicate_of,
                        "metadata": m.metadata,
                    }
                    for m in slack_messages
                ]
                with db.session_context() as bg_session:
                    db.save_cleaned_messages(bg_session, records, provider="slack")
            except Exception as exc:
                logger.warning("Failed to persist cleaned messages to DB: %s", exc)

        # Run RAG pipeline
        pipeline = Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
        count = pipeline.run(unique_messages)

        _slack_ingest_status[status_key]["chunks_stored"] = count
        _slack_ingest_status[status_key]["state"] = "done"
        logger.info(
            "Slack ingestion complete: %d messages fetched, %d unique, %d chunks stored.",
            len(all_raw_messages), len(unique_messages), count,
        )
    except Exception as exc:
        _slack_ingest_status[status_key]["state"] = "error"
        _slack_ingest_status[status_key]["error"] = str(exc)
        logger.exception("Slack ingestion failed: %s", exc)


@router.post(
    "/ingest/slack",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger Slack message ingestion",
)
def ingest_slack(
    background_tasks: BackgroundTasks,
    team_id: Optional[str] = Query(None, description="Slack team ID (optional if SLACK_BOT_TOKEN env is set)"),
    channel_id: Optional[str] = Query(None, description="Channel ID (leave empty to sync all accessible channels)"),
    limit_per_channel: int = Query(default=50, ge=1, le=200, description="Max messages per channel"),
    save_to_db: bool = Query(default=True, description="Persist cleaned messages to PostgreSQL"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Fetch Slack messages, clean, deduplicate, and run the RAG pipeline in the background."""
    _verify_team_ownership(session, user.id, team_id)
    background_tasks.add_task(_run_slack_ingest, team_id, channel_id, limit_per_channel, save_to_db)
    return {
        "status": "Slack ingestion started",
        "team_id": team_id,
        "channel_id": channel_id or "all",
        "limit_per_channel": limit_per_channel,
    }


@router.get(
    "/ingest/slack/status",
    summary="Check Slack ingestion progress",
)
def slack_ingest_status(
    team_id: Optional[str] = Query(None, description="Slack team ID"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Return the current Slack ingestion progress."""
    _verify_team_ownership(session, user.id, team_id)
    status_key = team_id or "default"
    status = _slack_ingest_status.get(status_key)
    if not status:
        return {"state": "idle"}
    return status


@router.get(
    "/slack/channels",
    summary="List available Slack channels",
)
def list_channels(
    team_id: Optional[str] = Query(None, description="Slack team ID"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Lists public and private channels the bot has access to."""
    _verify_team_ownership(session, user.id, team_id)
    try:
        loader = get_slack_loader_for_team(team_id=team_id)
        channels = loader.list_user_channels()
        return {"team_id": team_id, "channels_count": len(channels), "channels": channels}
    except Exception as exc:
        logger.error("Failed to list channels: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

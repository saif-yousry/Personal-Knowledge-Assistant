
"""
services/gmail_sync_service.py

File / Component Name: Gmail sync and backfill service.
What was added or modified: The historical backfill and checkpoint logic was tightened
so Gmail replies stay disabled while old emails are indexed. The post-backfill checkpoint
is saved before enabling forward sync, and the live catch-up run now starts from that
stored historyId instead of a separate "reply from now" shortcut.
Purpose / intent: Keep the pipeline strictly ordered as Backfill -> Checkpoint Save ->
Forward Catch-Up Sync so no automatic reply is sent for historical mail or for messages
that arrived while the backfill was still running.
"""

from __future__ import annotations

import logging
from functools import wraps
from threading import Lock, RLock
from datetime import datetime, timezone
from typing import Optional

from initializer import chunker, db, dispatcher, embedder, store
from loaders.gmail_loader import GmailLoader
from loaders.base_email_loader import EmailLoaderAuthRevokedError, EmailLoaderFetchError
from rag.pipeline import Pipeline
from services.gmail_autoresponder import gmail_autoresponder

logger = logging.getLogger(__name__)

_pipeline = lambda: Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
_sync_locks_guard = Lock()
_sync_locks: dict[int, RLock] = {}


def _serialize_user_sync(function):
    """Prevent sync, backfill, and start-now requests racing for one mailbox."""
    @wraps(function)
    def wrapper(user_id: int, *args, **kwargs):
        with _sync_locks_guard:
            user_lock = _sync_locks.setdefault(user_id, RLock())
        with user_lock:
            return function(user_id, *args, **kwargs)

    return wrapper


@_serialize_user_sync
def initialize_and_first_sync(
    user_id: int,
    google_creds: dict,
    label: str,
    status_dict: dict,
) -> None:
    """Called by POST /api/v1/ingest. Stores chosen label, initializes cursors,
    and fetches the first batch of historical emails without enabling replies."""
    status_dict[user_id] = {"state": "fetching", "fetched": 0, "chunks_stored": 0}
    start_history_id = None
    try:
        # (syncing checkpoint handeled)
        # Before a single historical page is scanned, clear any live checkpoint.
        # That keeps the scheduler fully silent while the backfill is still working.
        # Only the final post-backfill checkpoint is allowed to activate forwarding.
        # (auto responder added part by saif)
        with db.session_context() as session:
            db.update_google_history_id(session, user_id, None)
            db.update_google_backfill_cursor(session, user_id, None)
            db.update_google_sync_label(session, user_id, label)

        with GmailLoader(
            access_token=google_creds["access_token"],
            refresh_token=google_creds["refresh_token"],
            token_expiry=google_creds.get("token_expiry"),
            max_emails=60,
            label=label,
        ) as loader:
            # Phase 1: Hold checkpoint while backfill_cursor is active.
            # Persist the checkpoint with a non-null cursor before fetching the first
            # page, so later scheduled runs retain it while forwarding remains gated.
            # (auto responder added part by saif)
            start_history_id = loader.get_current_history_id()
            now = datetime.now(tz=timezone.utc)
            loader.before_timestamp = now
            with db.session_context() as session:
                db.update_google_backfill_cursor(session, user_id, now)
                db.update_google_history_id(session, user_id, start_history_id)
            emails = loader.fetch_emails()

        if loader.access_token != google_creds["access_token"]:
            with db.session_context() as session:
                db.update_google_access_token(session, user_id, loader.access_token, loader.token_expiry)

        status_dict[user_id]["fetched"] = len(emails)
        status_dict[user_id]["state"] = "processing"

        if emails:
            count = _pipeline().run(emails)
            status_dict[user_id]["chunks_stored"] = count
            new_cursor = min(e.date for e in emails)
        else:
            count = 0
            new_cursor = None

        with db.session_context() as session:
            db.update_google_backfill_cursor(session, user_id, new_cursor)

        if new_cursor is None:
            # Phase 2: Promote checkpoint and launch forward catch-up once
            # backfill_cursor becomes None. The checkpoint was persisted in Phase 1.
            # (auto responder added part by saif)
            logger.info(
                "Initial backfill complete for user %d; starting forward catch-up from historyId=%s.",
                user_id,
                start_history_id,
            )
            run_scheduled_sync(user_id, google_creds, status_dict)
            status_dict[user_id]["state"] = "ready"
        else:
            status_dict[user_id]["state"] = "backfilling"
        logger.info("Initialized sync for user %d: %d emails, cursor=%s.", user_id, count, new_cursor)

    except EmailLoaderAuthRevokedError:
        with db.session_context() as session:
            db.delete_google_credentials(session, user_id)
        status_dict[user_id]["state"] = "error"
        status_dict[user_id]["error"] = "Google access revoked. Please reconnect your Google account."
        logger.warning("Google credentials revoked for user %d. Credentials deleted.", user_id)

    except Exception as exc:
        status_dict[user_id]["state"] = "error"
        status_dict[user_id]["error"] = str(exc)
        logger.exception("Ingest initialization failed for user %d: %s", user_id, exc)


@_serialize_user_sync
def start_replies_from_now(
    user_id: int,
    google_creds: dict,
    label: str,
    status_dict: dict,
) -> str:
    """Enable Gmail replies from the current historyId without indexing old mail."""
    with GmailLoader(
        access_token=google_creds["access_token"],
        refresh_token=google_creds.get("refresh_token"),
        token_expiry=google_creds.get("token_expiry"),
        label=label,
    ) as loader:
        history_id = loader.get_current_history_id()
        access_token = loader.access_token
        token_expiry = loader.token_expiry

    with db.session_context() as session:
        if access_token != google_creds["access_token"]:
            db.update_google_access_token(session, user_id, access_token, token_expiry)
        db.update_google_sync_label(session, user_id, label)
        db.update_google_backfill_cursor(session, user_id, None)
        db.update_google_history_id(session, user_id, history_id)

    # Mark the mailbox active immediately; historical content indexing is optional.
    # (auto responder added part by saif)
    status_dict[user_id] = {
        "state": "ready",
        "mode": "auto_reply_from_now",
        "fetched": 0,
        "chunks_stored": 0,
    }
    logger.info(
        "Enabled auto-replies from historyId=%s for user %d without historical ingestion.",
        history_id,
        user_id,
    )
    return history_id


@_serialize_user_sync
def run_scheduled_sync(
    user_id: int,
    google_creds: dict,
    status_dict: Optional[dict] = None,
) -> None:
    """Called by the scheduler every 60s. Skips silently if user hasn't clicked ingest yet."""
    with db.session_context() as session:
        creds = db.find_google_credentials(session, user_id)

    if creds is None:
        return

    label = creds.sync_label
    backfill_cursor = creds.backfill_cursor
    history_id = creds.history_id

    # Phase 1: Hold checkpoint while backfill_cursor is active.
    # A history_id may already be stored, but it must not enable forwarding until
    # every historical page has been indexed.
    # (auto responder added part by saif)
    if backfill_cursor is not None:
        logger.info("Starting backfill for user %d, cursor=%s.", user_id, backfill_cursor)
        try:
            with GmailLoader(
                access_token=google_creds["access_token"],
                refresh_token=google_creds["refresh_token"],
                token_expiry=google_creds.get("token_expiry"),
                before_timestamp=backfill_cursor,
                max_emails=60,
                label=label,
            ) as loader:
                emails = loader.fetch_emails()

            if loader.access_token != google_creds["access_token"]:
                with db.session_context() as session:
                    db.update_google_access_token(session, user_id, loader.access_token, loader.token_expiry)
                google_creds = {**google_creds, "access_token": loader.access_token, "token_expiry": loader.token_expiry}

            if emails:
                chunks_stored = _pipeline().run(emails)
                new_cursor = min(e.date for e in emails)
                if status_dict is not None and user_id in status_dict:
                    status_dict[user_id]["fetched"] += len(emails)
                    status_dict[user_id]["chunks_stored"] += chunks_stored
                logger.info("Backfill: %d emails for user %d, cursor now %s.", len(emails), user_id, new_cursor)
            else:
                new_cursor = None
                logger.info("Backfill complete for user %d.", user_id)

            with db.session_context() as session:
                db.update_google_backfill_cursor(session, user_id, new_cursor)
            if new_cursor is None:
                # Phase 2: Promote checkpoint and launch forward catch-up once
                # backfill_cursor becomes None. Normal runs already have their
                # pre-fetch checkpoint in history_id; recover legacy incomplete
                # backfills from a fresh Gmail checkpoint if that value was lost.
                # (auto responder added part by saif)
                if history_id is None:
                    logger.warning(
                        "Backfill for user %d completed without a saved history checkpoint; "
                        "establishing a fresh checkpoint. Emails from the lost checkpoint window "
                        "cannot be recovered automatically.",
                        user_id,
                    )
                    with GmailLoader(
                        access_token=google_creds["access_token"],
                        refresh_token=google_creds.get("refresh_token"),
                        token_expiry=google_creds.get("token_expiry"),
                        label=label,
                    ) as loader:
                        history_id = loader.get_current_history_id()
                    if loader.access_token != google_creds["access_token"]:
                        with db.session_context() as session:
                            db.update_google_access_token(
                                session, user_id, loader.access_token, loader.token_expiry
                            )
                        google_creds = {
                            **google_creds,
                            "access_token": loader.access_token,
                            "token_expiry": loader.token_expiry,
                        }
                    with db.session_context() as session:
                        db.update_google_history_id(session, user_id, history_id)
                logger.info(
                    "Historical backfill complete for user %d; starting forward catch-up from historyId=%s.",
                    user_id,
                    history_id,
                )
                if status_dict is not None and user_id in status_dict:
                    status_dict[user_id]["state"] = "processing"
            else:
                return
        except EmailLoaderAuthRevokedError:
            with db.session_context() as session:
                db.delete_google_credentials(session, user_id)
            logger.warning("Google credentials revoked for user %d during backfill. Credentials deleted.", user_id)
            if status_dict is not None and user_id in status_dict:
                status_dict[user_id]["state"] = "error"
                status_dict[user_id]["error"] = "Google access revoked. Please reconnect your Google account."
            return
        except Exception as exc:
            logger.exception("Backfill failed for user %d: %s", user_id, exc)
            if status_dict is not None and user_id in status_dict:
                status_dict[user_id]["state"] = "error"
                status_dict[user_id]["error"] = str(exc)
            return

    if history_id is None:
        logger.info("Forward sync is not ready for user %d; no history checkpoint exists.", user_id)
        return

    # (syncing checkpoint handeled)
    # Pass 2: Gmail historyId identifies newly added messages; do not use timestamps here.
    # (auto responder added part by saif)
    logger.info("Starting forward sync for user %d, historyId=%s.", user_id, history_id)
    try:
        with GmailLoader(
            access_token=google_creds["access_token"],
            refresh_token=google_creds["refresh_token"],
            token_expiry=google_creds.get("token_expiry"),
            label=label,
        ) as loader:
            try:
                new_emails, new_history_id = loader.fetch_new_by_history(history_id)
            except EmailLoaderFetchError:
                new_history_id = loader.get_current_history_id()
                new_emails = []
                logger.warning("historyId expired for user %d, reset to %s.", user_id, new_history_id)

        if loader.access_token != google_creds["access_token"]:
            with db.session_context() as session:
                db.update_google_access_token(session, user_id, loader.access_token, loader.token_expiry)

        if new_emails:
            checkpoint_blocked = False
            for email in new_emails:
                result = gmail_autoresponder.process_email(email, user_id=user_id)
                logger.info(
                    "Forward sync processed Gmail message %s for user %d (result=%s).",
                    email.id,
                    user_id,
                    result.get("status"),
                )
                checkpoint_blocked = checkpoint_blocked or result.get("status") == "processing"
        else:
            checkpoint_blocked = False
            logger.info("Forward sync: no new emails for user %d.", user_id)

        if checkpoint_blocked:
            logger.info(
                "Holding Gmail history checkpoint for user %d while a message is processing.",
                user_id,
            )
        else:
            with db.session_context() as session:
                db.update_google_history_id(session, user_id, new_history_id)

        if status_dict is not None and user_id in status_dict:
            status_dict[user_id]["state"] = "processing" if checkpoint_blocked else "ready"

    except EmailLoaderAuthRevokedError:
        with db.session_context() as session:
            db.delete_google_credentials(session, user_id)
        logger.warning("Google credentials revoked for user %d during forward sync. Credentials deleted.", user_id)
    except Exception as exc:
        logger.exception("Forward sync failed for user %d: %s", user_id, exc)

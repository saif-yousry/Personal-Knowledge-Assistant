"""
routers/gmail_ingest.py

File / Component Name: Gmail ingestion API routes.
What was added or modified: The historical-ingestion endpoint is kept as the only
allowed trigger for Gmail backfill, and the shortcut that starts replies from the
current moment has been disabled so it cannot bypass the backfill-first sequence.
Purpose / intent: Enforce the strict pipeline of Backfill -> Checkpoint Save ->
Forward Catch-Up Sync.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from initializer import db
from services.app_auth_service import get_current_user
from services.gmail_sync_service import initialize_and_first_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ingest"])

# In-memory ingestion progress tracker keyed by user_id
ingest_status_by_user: dict[int, dict] = {}


class AutoResponderSettingsRequest(BaseModel):
    enabled: bool


@router.get(
    "/email/auto-reply/settings",
    summary="Get Gmail autoresponder setting",
)
def get_auto_responder_settings(
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    credentials = db.find_google_credentials(session, user.id)
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account not linked. Connect Google first.",
        )
    return {"enabled": credentials.autoresponder_enabled}


@router.put(
    "/email/auto-reply/settings",
    summary="Enable or disable Gmail autoresponder",
)
def update_auto_responder_settings(
    payload: AutoResponderSettingsRequest = Body(...),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    if not db.set_google_autoresponder_enabled(session, user.id, payload.enabled):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account not linked. Connect Google first.",
        )
    return {"enabled": payload.enabled}


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger email ingestion",
    description="Initializes sync cursors and fetches the first batch of emails in the background. The scheduler continues automatically after this.",
)
def ingest(
    background_tasks: BackgroundTasks,
    label: str = Query(default="INBOX", description="Gmail label/folder to sync"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Initialize email sync for a user and fetch the first batch in the background."""
    google_creds = db.find_google_credentials(session, user.id)
    if not google_creds:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google account not linked. Connect Google first.",
        )

    background_tasks.add_task(
        initialize_and_first_sync,
        user.id,
        {"access_token": google_creds.access_token, "refresh_token": google_creds.refresh_token, "token_expiry": google_creds.token_expiry},
        label,
        ingest_status_by_user,
    )
    return {"status": "Ingestion started", "label": label}


@router.get(
    "/ingest/status",
    summary="Check ingestion progress",
    description="Returns the current ingestion status for the logged-in user.",
)
def ingest_status(
    user=Depends(get_current_user),
):
    """Return the current ingestion progress for the logged-in user."""
    status = ingest_status_by_user.get(user.id)
    if not status:
        return {"state": "idle"}
    return status


@router.post(
    "/email/auto-reply/start",
    status_code=status.HTTP_400_BAD_REQUEST,
    summary="Disabled shortcut: start Gmail auto-replies from now",
    description=(
        "Deprecated. Historical ingestion must complete first and only then the system "
        "may start forward catch-up replies. This endpoint intentionally blocks the "
        "bypass path so users cannot skip the required backfill checkpoint."
    ),
)
def start_auto_replies(
    label: str = Query(default="INBOX", description="Gmail label/folder to monitor"),
    user=Depends(get_current_user),
    session: Session = Depends(db.get_session),
):
    """Reject the bypass flow that starts replies before backfill completion."""
    # This shortcut violates the required ordering: Backfill -> Checkpoint Save ->
    # Forward Catch-Up Sync. We keep it present only as an explicit blocker so the UI
    # and any callers cannot start live replies before the historical import is done.
    # (auto responder added part by saif)
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Auto-replies cannot start from 'now' because historical backfill must finish first. Use the ingestion flow and wait for the checkpoint to be saved.",
    )

"""
routers/email_agent.py

Reason: HTTP API endpoint to receive incoming emails (from external webhooks,
mail services, or testing clients), trigger agentic reasoning and email dispatch,
and persist the resulting email exchange into the vector store.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from initializer import db
from schemas.email_agent import IncomingEmailRequest, ProcessEmailResponse
from services.app_auth_service import verify_access_token
from services.email_agent_service import process_incoming_email

logger = logging.getLogger(__name__)

# Reason: Prefix /api/v1/email groups email-automation endpoints under versioned API routes.
router = APIRouter(prefix="/api/v1/email", tags=["email-agent"])
optional_oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,
)


# Reason: Helper to optionally authenticate a user if an Authorization token is provided,
# allowing both authenticated users and unauthenticated webhook callbacks to invoke the endpoint.
async def get_optional_user(
    token: Optional[str] = Depends(optional_oauth2_scheme),
    session: Session = Depends(db.get_session),
):
    """Optional auth dependency allowing webhook access while recognizing logged-in users."""
    if token is None:
        return None

    try:
        user_id = verify_access_token(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user = db.find_user_by_id(session, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user


# Reason: Core POST endpoint to receive an incoming email.
# The incoming email is parsed for metadata and body, sent to the agent in structured format
# (auto responder added part by saif) Passes this request to the shared prompt and
# email-only reply tool; successful exchanges are stored in the vector database.
@router.post(
    "/incoming",
    response_model=ProcessEmailResponse,
    status_code=status.HTTP_200_OK,
    summary="Process incoming email via agentic reasoning",
    description=(
        "Parses an incoming email, passes sender, subject, and body in structured format "
        "to the agent, allows the agent to reason, retrieve vector store facts if needed, "
        "and send a reply. Upon successful sending, both emails are stored in the vector store."
    ),
)
def handle_incoming_email(
    payload: IncomingEmailRequest,
    user=Depends(get_optional_user),
):
    """
    Handle an incoming email by triggering the automated email agent workflow.
    """
    try:
        user_id = user.id if user and hasattr(user, "id") else None
        result = process_incoming_email(payload, user_id=user_id)
        return ProcessEmailResponse(
            status=result["status"],
            incoming_email_id=result["incoming_email_id"],
            sender=result["sender"],
            subject=result["subject"],
            email_sent=result["email_sent"],
            sent_details=result.get("sent_details"),
            reply_email_id=result.get("reply_email_id"),
            chunks_stored=result.get("chunks_stored", 0),
            agent_reply=result.get("agent_reply", ""),
        )
    except Exception as exc:
        logger.exception("Failed to process incoming email: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing incoming email: {exc}",
        )

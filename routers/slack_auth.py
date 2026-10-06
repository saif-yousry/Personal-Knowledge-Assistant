"""
routers/slack_auth.py

FastAPI router exposing Slack OAuth 2.0 install, callback, health, and
revoke endpoints.

Endpoints
---------
GET  /slack/install          — Start the Slack OAuth install flow.
GET  /slack/oauth/callback   — Handle the callback from Slack.
GET  /slack/health           — Liveness check for Slack auth config.
DELETE /slack/revoke/{team_id} — Revoke and delete stored token.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, RedirectResponse

from config import settings
from services.app_auth_service import get_current_user
from services.slack_auth_service import (
    exchange_code_for_token,
    generate_install_url,
    revoke_token,
    validate_state,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/slack", tags=["Slack OAuth"])


# ---------------------------------------------------------------------------
# GET /slack/install
# ---------------------------------------------------------------------------

@router.get(
    "/install",
    summary="Start Slack OAuth install flow",
)
def slack_install(
    user=Depends(get_current_user),
):
    """Return the Slack OAuth authorisation URL (frontend navigates to it)."""
    try:
        install_url, state = generate_install_url(user.id)
    except RuntimeError as exc:
        logger.error("Failed to generate install URL: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "install_url": install_url,
            "state": state,
            "message": "Redirect your user to `install_url` to begin the Slack OAuth flow.",
        }
    )


# ---------------------------------------------------------------------------
# GET /slack/oauth/callback
# ---------------------------------------------------------------------------

@router.get(
    "/oauth/callback",
    summary="Handle Slack OAuth callback",
)
async def slack_oauth_callback(
    code: Optional[str] = Query(None, description="Temporary OAuth code from Slack."),
    state: Optional[str] = Query(None, description="CSRF state token echoed back by Slack."),
    error: Optional[str] = Query(None, description="Error code if the user denied access."),
):
    """OAuth 2.0 callback handler: validate state, exchange code, persist tokens."""
    if error:
        logger.warning("Slack OAuth denied by user: %s", error)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Slack OAuth was not completed. Reason: {error}",
        )

    if not code:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'code' query parameter.")

    if not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'state' parameter. Please restart the install flow from /slack/install.",
        )

    user_id = validate_state(state)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OAuth state. Please restart the install flow from /slack/install.",
        )

    try:
        token_bundle = await exchange_code_for_token(code, user_id)
    except ValueError as exc:
        logger.error("Token exchange failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during token exchange.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token exchange failed unexpectedly.") from exc

    team_id = token_bundle["team_id"]
    team_name = token_bundle["team_name"]
    logger.info("Slack app successfully installed for team_id=%s (%s)", team_id, team_name)

    success_url = getattr(settings, "SLACK_INSTALL_SUCCESS_URL", "")
    if success_url:
        return RedirectResponse(url=success_url, status_code=status.HTTP_303_SEE_OTHER)

    return RedirectResponse(
        url=f"/?slack_team_id={team_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


# ---------------------------------------------------------------------------
# GET /slack/health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Slack auth health check",
)
def slack_health():
    """Liveness / readiness check for the Slack auth module."""
    issues: list[str] = []

    if not settings.SLACK_CLIENT_ID:
        issues.append("SLACK_CLIENT_ID is not set.")
    if not settings.SLACK_CLIENT_SECRET.get_secret_value():
        issues.append("SLACK_CLIENT_SECRET is not set.")
    if not settings.SLACK_REDIRECT_URI:
        issues.append("SLACK_REDIRECT_URI is not set (using default localhost).")

    env_bot_token_set = bool(settings.SLACK_BOT_TOKEN.get_secret_value())
    health_status = "degraded" if issues else "ok"

    return JSONResponse(
        content={
            "status": health_status,
            "env_bot_token_present": env_bot_token_set,
            "issues": issues,
        },
        status_code=status.HTTP_200_OK if health_status == "ok" else status.HTTP_207_MULTI_STATUS,
    )


# ---------------------------------------------------------------------------
# DELETE /slack/revoke/{team_id}
# ---------------------------------------------------------------------------

@router.delete(
    "/revoke/{team_id}",
    summary="Revoke stored token for a workspace",
)
def slack_revoke(team_id: str, user=Depends(get_current_user)):
    """Revoke and delete the stored Slack token for a workspace."""
    try:
        revoke_token(team_id)
    except Exception as exc:
        logger.exception("Failed to revoke token for team_id=%s", team_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "ok": True,
            "message": f"Token revocation requested for team_id='{team_id}'.",
        }
    )

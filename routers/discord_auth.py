"""
routers/discord_auth.py

FastAPI router for Discord OAuth 2.0 bot installation flow.

Endpoints
---------
GET  /discord/authorize   — Redirect to Discord OAuth authorize page.
GET  /discord/callback    — Handle the OAuth callback from Discord.
GET  /discord/health      — Liveness check for Discord auth config.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse

from config import settings
from services.app_auth_service import get_current_user
from services.discord_auth_service import (
    DiscordIntegrationError,
    build_authorization_url,
    exchange_code_for_tokens,
    get_discord_user,
    save_connection,
    verify_bot_guild_access,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/discord", tags=["Discord OAuth"])

STATE_COOKIE_NAME = "discord_oauth_state"
STATE_COOKIE_MAX_AGE = 600

# Maps OAuth state token → app user_id (set in /authorize, read in /callback)
_state_user_map: dict[str, int] = {}


def _state_cookie_is_secure() -> bool:
    return bool(settings.DISCORD_REDIRECT_URI and settings.DISCORD_REDIRECT_URI.startswith("https://"))


def _error_response(
    status_code: int, detail: str, *, clear_state: bool = False
) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    if clear_state:
        response.delete_cookie(
            STATE_COOKIE_NAME, secure=_state_cookie_is_secure(), samesite="lax"
        )
    return response


@router.get(
    "/authorize",
    summary="Start Discord OAuth install flow",
)
def discord_authorize(user=Depends(get_current_user)) -> JSONResponse:
    """Return the Discord OAuth authorization URL (frontend navigates to it)."""
    state = secrets.token_urlsafe(32)
    _state_user_map[state] = user.id

    try:
        authorization_url = build_authorization_url(state)
    except DiscordIntegrationError as exc:
        _state_user_map.pop(state, None)
        return _error_response(exc.status_code, str(exc))

    response = JSONResponse(content={"authorization_url": authorization_url})
    response.set_cookie(
        key=STATE_COOKIE_NAME,
        value=state,
        max_age=STATE_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=_state_cookie_is_secure(),
    )
    return response


@router.get(
    "/callback",
    summary="Handle Discord OAuth callback",
)
def discord_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    guild_id: str | None = Query(default=None),
    permissions: str | None = Query(default=None),
) -> JSONResponse:
    """OAuth 2.0 callback: validate state, exchange code, save connection."""
    stored_state = request.cookies.get(STATE_COOKIE_NAME)
    if not stored_state or not state:
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord OAuth state is missing.")
    if not secrets.compare_digest(stored_state, state):
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord OAuth state is invalid.")
    if error:
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord authorization was denied.", clear_state=True)
    if not code:
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord authorization code is missing.", clear_state=True)
    if permissions is not None and not permissions.isdigit():
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord permissions are invalid.", clear_state=True)

    app_user_id = _state_user_map.pop(state, None)
    if app_user_id is None:
        return _error_response(status.HTTP_400_BAD_REQUEST, "Discord OAuth state has no associated user.", clear_state=True)

    try:
        token_data = exchange_code_for_tokens(code)
        token_guild_id = token_data.get("guild_id")
        if not isinstance(token_guild_id, str) or not token_guild_id:
            return _error_response(status.HTTP_400_BAD_REQUEST, "Discord guild is missing.", clear_state=True)
        if guild_id and token_guild_id != guild_id:
            return _error_response(
                status.HTTP_400_BAD_REQUEST, "Discord guild information is inconsistent.", clear_state=True
            )

        discord_user_id = get_discord_user(token_data["access_token"])
        verify_bot_guild_access(token_guild_id)
        save_connection(
            user_id=app_user_id,
            discord_user_id=discord_user_id,
            guild_id=token_guild_id,
            token_data=token_data,
            granted_permissions=permissions,
        )
    except DiscordIntegrationError as exc:
        return _error_response(exc.status_code, str(exc), clear_state=True)

    response = RedirectResponse(
        url=f"/?discord_guild_id={token_guild_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )
    response.delete_cookie(STATE_COOKIE_NAME, secure=_state_cookie_is_secure(), samesite="lax")
    return response


@router.get(
    "/health",
    summary="Discord auth health check",
)
def discord_health():
    """Liveness / readiness check for the Discord auth module."""
    issues: list[str] = []

    if not settings.DISCORD_CLIENT_ID:
        issues.append("DISCORD_CLIENT_ID is not set.")
    if not settings.DISCORD_CLIENT_SECRET.get_secret_value():
        issues.append("DISCORD_CLIENT_SECRET is not set.")
    if not settings.DISCORD_REDIRECT_URI:
        issues.append("DISCORD_REDIRECT_URI is not set.")
    if not settings.DISCORD_BOT_TOKEN.get_secret_value():
        issues.append("DISCORD_BOT_TOKEN is not set.")

    health_status = "degraded" if issues else "ok"

    return JSONResponse(
        content={
            "status": health_status,
            "bot_token_present": bool(settings.DISCORD_BOT_TOKEN.get_secret_value()),
            "issues": issues,
        },
        status_code=status.HTTP_200_OK if health_status == "ok" else status.HTTP_207_MULTI_STATUS,
    )

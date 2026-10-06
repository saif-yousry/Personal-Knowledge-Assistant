"""
Discord OAuth 2.0 token management service.

Responsibilities:
  - Build the Discord OAuth authorization URL.
  - Exchange the temporary code for tokens.
  - Persist/retrieve tokens via the app's Database singleton.
  - Verify bot guild access.
  - Fetch the authenticated Discord user ID.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from config import settings

logger = logging.getLogger(__name__)

DISCORD_API_BASE_URL = "https://discord.com/api/v10"
DISCORD_AUTHORIZE_URL = "https://discord.com/oauth2/authorize"
DISCORD_BOT_PERMISSIONS = settings.DISCORD_BOT_PERMISSIONS
DISCORD_OAUTH_SCOPES = settings.DISCORD_BOT_SCOPES

DISCORD_CLIENT_ID = settings.DISCORD_CLIENT_ID
DISCORD_REDIRECT_URI = settings.DISCORD_REDIRECT_URI


class DiscordIntegrationError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _get_db():
    from initializer import db
    return db


def _require_oauth_config() -> tuple[str, str, str]:
    client_secret = settings.DISCORD_CLIENT_SECRET.get_secret_value()
    if not all((DISCORD_CLIENT_ID, client_secret, DISCORD_REDIRECT_URI)):
        raise DiscordIntegrationError("Discord configuration is incomplete.", 500)
    return DISCORD_CLIENT_ID, client_secret, DISCORD_REDIRECT_URI


def _require_bot_token() -> str:
    bot_token = settings.DISCORD_BOT_TOKEN.get_secret_value()
    if not bot_token:
        raise DiscordIntegrationError("DISCORD_BOT_TOKEN is not set.", 500)
    return bot_token


def _parse_token_payload(payload: dict[str, Any]) -> dict[str, Any]:
    access_token = payload.get("access_token")
    token_type = payload.get("token_type")
    expires_in = payload.get("expires_in")
    scopes = payload.get("scope")

    if not all(isinstance(v, str) and v for v in (access_token, token_type, scopes)):
        raise DiscordIntegrationError("Discord token response was invalid.")
    if isinstance(expires_in, bool) or not isinstance(expires_in, (int, float)) or expires_in <= 0:
        raise DiscordIntegrationError("Discord token response was invalid.")

    refresh_token = payload.get("refresh_token")
    guild_id: str | None = None
    guild = payload.get("guild")
    if isinstance(guild, dict) and isinstance(guild.get("id"), str):
        guild_id = guild["id"]

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": token_type,
        "scopes": scopes,
        "expires_at": datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        "guild_id": guild_id,
    }


def build_authorization_url(state: str) -> str:
    if not state:
        raise DiscordIntegrationError("OAuth state is required.", 400)
    client_id, _, redirect_uri = _require_oauth_config()
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": DISCORD_OAUTH_SCOPES,
        "permissions": str(DISCORD_BOT_PERMISSIONS),
        "state": state,
    }
    return f"{DISCORD_AUTHORIZE_URL}?{urlencode(params)}"


def exchange_code_for_tokens(code: str) -> dict[str, Any]:
    if not code:
        raise DiscordIntegrationError("Discord authorization code is required.", 400)
    client_id, client_secret, redirect_uri = _require_oauth_config()

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{DISCORD_API_BASE_URL}/oauth2/token",
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.HTTPError as exc:
        raise DiscordIntegrationError("Discord token exchange failed.") from exc

    if response.is_error:
        logger.error("Discord token exchange HTTP %s: %s", response.status_code, response.text)
        raise DiscordIntegrationError("Discord token exchange failed.")

    payload = response.json()
    logger.info("Discord token exchange response keys: %s", list(payload.keys()))
    return _parse_token_payload(payload)


def get_discord_user(access_token: str) -> str:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{DISCORD_API_BASE_URL}/users/@me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as exc:
        raise DiscordIntegrationError("Discord user lookup failed.") from exc

    if response.is_error:
        raise DiscordIntegrationError("Discord user lookup failed.")

    payload = response.json()
    user_id = payload.get("id") if isinstance(payload, dict) else None
    if not isinstance(user_id, str) or not user_id:
        raise DiscordIntegrationError("Discord user response was invalid.")
    return user_id


def verify_bot_guild_access(guild_id: str) -> None:
    bot_token = _require_bot_token()
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(
                f"{DISCORD_API_BASE_URL}/guilds/{guild_id}",
                headers={"Authorization": f"Bot {bot_token}"},
            )
    except httpx.HTTPError as exc:
        raise DiscordIntegrationError("Discord guild verification failed.") from exc

    if response.status_code in (401, 403, 404):
        raise DiscordIntegrationError("Discord bot cannot access the selected guild.", 403)
    if response.is_error:
        raise DiscordIntegrationError("Discord guild verification failed.")


def save_connection(
    *,
    user_id: int,
    discord_user_id: str,
    guild_id: str,
    token_data: dict[str, Any],
    granted_permissions: str | None = None,
) -> None:
    db = _get_db()
    with db.session_context() as session:
        db.save_discord_credentials(
            session,
            user_id=user_id,
            discord_user_id=discord_user_id,
            guild_id=guild_id,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_type=token_data.get("token_type", "Bearer"),
            scopes=token_data.get("scopes", "identify bot"),
            expires_at=token_data.get("expires_at"),
            granted_permissions=granted_permissions,
        )
    logger.info("Saved Discord connection for user=%s guild=%s", discord_user_id, guild_id)


def get_bot_token() -> str:
    return _require_bot_token()

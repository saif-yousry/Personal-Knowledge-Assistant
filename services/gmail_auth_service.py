"""
services/auth_service.py

AuthService — the Google OAuth 2.0 domain logic. Everything that is not
HTTP handling lives here:

- Build the Google authorization URL (with CSRF state).
- Validate the state echoed back on the callback.
- Exchange the authorization code for tokens.
- Fetch the Google user profile.
- Create or update the user row in PostgreSQL (via database module).
- Refresh expired access tokens.

The route layer translates between HTTP and this service.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import Settings, settings as app_settings
from database.models import User

logger = logging.getLogger(__name__)

# How long a generated OAuth `state` remains valid before it is rejected.
STATE_TTL_SECONDS = 600  # 10 minutes


class AuthError(Exception):
    """Base class for all auth-flow failures."""


class StateValidationError(AuthError):
    """The OAuth `state` parameter was missing, stale, or unknown."""


class OAuthExchangeError(AuthError):
    """Code->token exchange or profile fetch failed."""


class TokenRefreshError(AuthError):
    """Could not refresh an expired access token."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuthService:
    """Google OAuth 2.0 domain logic: authorization, token exchange, and user persistence."""

    def __init__(self, settings: Optional[Settings] = None, db=None) -> None:
        self.settings = settings or app_settings
        self.oauth_flows = {}
        if db is None:
            from initializer import db  # pylint: disable=import-outside-toplevel
        self.db = db

    # ------------------------------------------------------------------
    # OAuth URL
    # ------------------------------------------------------------------

    def get_authorization_url(self, user_id: int) -> tuple[str, str]:
        """
        Build the Google authorization page URL.

        Returns (authorization_url, state). The `user_id` is stored alongside
        the flow so the callback knows which registered user to link.
        """
        flow = Flow.from_client_config(
            self.settings.google_client_config, scopes=self.settings.google_scopes
        )
        flow.redirect_uri = self.settings.GOOGLE_REDIRECT_URI

        authorization_url, state = flow.authorization_url(
            access_type="offline",          # request a refresh_token
            prompt="consent",               # force consent so refresh_token is re-issued
            include_granted_scopes="true",
        )
        # save the flow with its creation time and user_id
        self.oauth_flows[state] = (flow, time.time(), user_id)
        logger.info("Stored states: %s",list(self.oauth_flows.keys()))
        logger.info("Generated OAuth authorization URL with state=%s...", state[:8])
        return authorization_url, state

    # ------------------------------------------------------------------
    # Callback / authentication
    # ------------------------------------------------------------------

    def authenticate(self, session, code: str, state: str) -> User:
        """
        Full callback handling: validate state, exchange the code, fetch the
        Google profile, and link credentials to the user. Returns the user.
        """
        entry = self.oauth_flows.pop(state, None)

        if entry is None:
            raise StateValidationError("Flow not found")

        flow, created_at, user_id = entry
        if time.time() - created_at > STATE_TTL_SECONDS:
            raise StateValidationError("OAuth state has expired.")

        logger.info("Flow found")
        logger.info("Redirect URI: %s", flow.redirect_uri)

        logger.info("Starting token exchange")

        try:
            flow.oauth2session.scope = None
            flow.fetch_token(code=code)

        except Exception as exc:
            logger.exception("Token exchange failed")
            raise OAuthExchangeError(
                f"Failed to exchange authorization code: {exc}"
            ) from exc

        credentials = flow.credentials
        profile = self._get_profile(credentials)
        return self._save_user(session, user_id, profile, credentials)

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def refresh_access_token(self, refresh_token: str) -> tuple[str, datetime]:
        """Exchange a stored refresh_token for a fresh access token."""
        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self.settings.GOOGLE_CLIENT_ID,
            client_secret=self.settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
        )
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            logger.error("Token refresh failed: %s", exc)
            raise TokenRefreshError(f"Failed to refresh access token: {exc}") from exc
        if not creds.token:
            raise TokenRefreshError("Token refresh returned no access token.")
        return creds.token, creds.expiry

    # ------------------------------------------------------------------
    # Google profile + persistence
    # ------------------------------------------------------------------

    def _get_profile(self, credentials: Credentials) -> dict[str, Any]:
        """Fetch identity info (id, email, name, picture) for the token holder."""
        try:
            service = build("oauth2", "v2", credentials=credentials, cache_discovery=False)
            profile = service.userinfo().get().execute()
        except HttpError as exc:
            logger.error("Profile fetch failed: %s", exc)
            raise OAuthExchangeError(f"Failed to fetch Google profile: {exc}") from exc
        return profile

    def _save_user(self, session, user_id: int, profile: dict[str, Any], credentials: Credentials) -> User:
        """
        Link Google credentials to the logged-in user identified by `user_id`.
        """
        google_id = str(profile.get("id") or "").strip()

        if not google_id:
            raise OAuthExchangeError(
                "Google profile is missing required identity field (id)."
            )

        access_token = credentials.token
        refresh_token = credentials.refresh_token
        token_expiry = credentials.expiry
        if token_expiry is not None and token_expiry.tzinfo is None:
            token_expiry = token_expiry.replace(tzinfo=timezone.utc)

        try:
            self.db.save_google_credentials(
                session,
                user_id,
                google_id=google_id,
                access_token=access_token,
                refresh_token=refresh_token,
                token_expiry=token_expiry,
            )
        except ValueError as exc:
            raise OAuthExchangeError(str(exc)) from exc

        return self.db.find_user_by_id(session, user_id)

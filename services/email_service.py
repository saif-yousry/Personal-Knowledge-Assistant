"""
services/email_service.py

Reason: Provides email delivery through Gmail API using the user's OAuth tokens.
This abstraction decouples Gmail API delivery from the agent tool while tracking
sent emails for downstream verification and RAG vector store persistence.
"""

from __future__ import annotations

import base64
import logging
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set by the agent controller before each tool-execution round.
# The email reply tool reads this so the LLM never sees user_id.
_current_user_id: ContextVar[Optional[int]] = ContextVar("_current_user_id", default=None)
# Keep the incoming sender outside model-controlled tool arguments.
# (auto responder added part by saif)
_current_email_reply_to: ContextVar[Optional[str]] = ContextVar(
    "_current_email_reply_to",
    default=None,
)
_current_sent_email: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "_current_sent_email", default=None
)


# Reason: Exception class to capture and handle email delivery errors gracefully
# without crashing the agent execution loop.
class EmailDeliveryError(Exception):
    """Raised when sending an email fails."""


# Reason: Centralized service to manage email dispatching and sent-message tracking.
class EmailService:
    """
    Sends messages only through the authenticated user's Gmail API connection.
    """

    def __init__(self) -> None:
        # Reason: In-memory store of sent emails so the calling service can inspect
        # whether an email was sent successfully and retrieve exact parameters.
        self._sent_emails: list[dict[str, Any]] = []

    # Reason: Core dispatch method for sending through the linked Gmail account.
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        """
        Send an email to `to` with `subject` and `body`.
        user_id is resolved from the _current_user_id ContextVar set by the
        controller — callers no longer pass it explicitly.
        Returns a dict containing delivery details and a unique message ID.
        """
        user_id: Optional[int] = _current_user_id.get()
        if user_id is None:
            raise EmailDeliveryError(
                "Gmail API delivery requires an authenticated app user with a linked Google account."
            )

        try:
            from initializer import db

            with db.session_context() as session:
                creds_row = db.find_google_credentials(session, user_id)
        except Exception as exc:
            logger.exception("Failed to look up Gmail credentials for user_id=%s", user_id)
            raise EmailDeliveryError(f"Gmail credential lookup failed: {exc}") from exc

        if creds_row is None:
            raise EmailDeliveryError(
                "No Gmail account is linked to this user. Connect Google before sending email."
            )

        try:
            return self._send_via_gmail_api(
                creds_row=creds_row,
                to=to,
                subject=subject,
                body=body,
            )
        except EmailDeliveryError:
            raise
        except Exception as exc:
            logger.exception("Gmail delivery failed to %s", to)
            raise EmailDeliveryError(f"Gmail delivery failed: {exc}") from exc

    # Reason: Helper to send emails via the official Gmail API using OAuth credentials.
    def _send_via_gmail_api(
        self,
        creds_row: Any,
        to: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        """Send email using Google API client with user OAuth credentials."""
        expiry = creds_row.token_expiry
        if expiry is not None and expiry.tzinfo is not None:
            expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
        creds = Credentials(
            token=creds_row.access_token,
            refresh_token=creds_row.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
            scopes=settings.google_scopes,
            expiry=expiry,
        )
        if not creds.valid:
            if not creds.refresh_token:
                raise EmailDeliveryError(
                    "Gmail access token expired and no refresh token is available. "
                    "Reconnect the Google account."
                )
            creds.refresh(Request())
            from initializer import db

            with db.session_context() as session:
                db.update_google_access_token(
                    session, creds_row.user_id, creds.token, creds.expiry
                )

        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        try:
            profile = service.users().getProfile(userId="me").execute()
            account_email = profile.get("emailAddress")
            if not account_email:
                raise EmailDeliveryError("Gmail did not return the linked account address.")
            message = MIMEText(body, "plain", "utf-8")
            message["to"] = to
            message["from"] = account_email
            message["subject"] = subject

            raw_payload = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
            sent_response = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw_payload})
                .execute()
            )
        finally:
            try:
                service.close()
            except Exception:
                logger.warning("Failed to close Gmail API service.", exc_info=True)

        msg_id = sent_response.get("id", str(uuid.uuid4()))
        record = {
            "status": "sent",
            "provider": "gmail_api",
            "message_id": msg_id,
            "to": to,
            "from": account_email,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._record_sent_email(record)
        logger.info("Email sent via Gmail API to %s (id=%s)", to, msg_id)
        return record

    def _record_sent_email(self, record: dict[str, Any]) -> None:
        self._sent_emails.append(record)
        _current_sent_email.set(record)

    def clear_current_sent_email(self) -> None:
        """Clear the sent-email record for the current agent execution context."""
        _current_sent_email.set(None)

    def get_current_sent_email(self) -> Optional[dict[str, Any]]:
        """Return the email sent by the current agent execution, if any."""
        return _current_sent_email.get()

    # Reason: Retrieve the most recent email sent to facilitate response inspection.
    def get_last_sent_email(self) -> Optional[dict[str, Any]]:
        """Return the most recently sent email record, or None if none sent yet."""
        if not self._sent_emails:
            return None
        return self._sent_emails[-1]

    # Reason: Retrieve list of all sent emails for testing or audit tracking.
    def get_sent_emails(self) -> list[dict[str, Any]]:
        """Return all sent email records."""
        return list(self._sent_emails)

    # Reason: Clear recorded sent emails history (useful between tests/sessions).
    def clear_history(self) -> None:
        """Clear recorded email history."""
        self._sent_emails.clear()


# Reason: Global singleton instance of EmailService used across tools and services.
email_service = EmailService()

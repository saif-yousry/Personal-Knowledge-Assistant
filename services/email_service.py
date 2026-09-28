"""
services/email_service.py

Reason: Provides email delivery logic (supporting Gmail API via user OAuth tokens,
standard SMTP servers).
This abstraction decouples email delivery protocols from the agent tool while tracking
sent emails for downstream verification and RAG vector store persistence.
"""

from __future__ import annotations

import base64
import email.message
import logging
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any, Optional

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Reason: Exception class to capture and handle email delivery errors gracefully
# without crashing the agent execution loop.
class EmailDeliveryError(Exception):
    """Raised when sending an email fails."""


# Reason: Centralized service to manage email dispatching and sent-message tracking.
class EmailService:
    """
    Handles sending emails across multiple backends:
      1. Gmail API (when Google OAuth credentials exist for the user)
      2. SMTP (when SMTP server host is configured in settings)
    """

    def __init__(self) -> None:
        # Reason: In-memory store of sent emails so the calling service can inspect
        # whether an email was sent successfully and retrieve exact parameters.
        self._sent_emails: list[dict[str, Any]] = []

    # Reason: Core dispatch method to send an email, choosing the best available transport.
    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        sender: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Send an email to `to` with `subject` and `body`.
        Returns a dict containing delivery details and a unique message ID.
        """
        from_email = sender or settings.DEFAULT_SENDER_EMAIL
        failures: list[str] = []

        # Attempt 1: Try Gmail API if a user_id is provided and credentials exist
        if user_id is not None:
            try:
                from initializer import db
                creds_row = db.find_google_credentials(user_id)
            except Exception as exc:
                logger.exception("Failed to look up Gmail credentials for user_id=%s", user_id)
                failures.append(f"Gmail credential lookup failed: {exc}")
            else:
                if creds_row:
                    try:
                        return self._send_via_gmail_api(
                            creds_row=creds_row,
                            to=to,
                            subject=subject,
                            body=body,
                            from_email=from_email,
                        )
                    except Exception as exc:
                        logger.exception("Gmail delivery failed to %s", to)
                        failures.append(f"Gmail delivery failed: {exc}")
                else:
                    failures.append("No Gmail credentials are linked to this user")
                    logger.warning("No Gmail credentials are linked to user_id=%s", user_id)
        else:
            failures.append("No authenticated user ID was provided for Gmail delivery")

        # Attempt 2: Try SMTP if SMTP_HOST is configured
        if settings.SMTP_HOST:
            try:
                return self._send_via_smtp(
                    to=to,
                    subject=subject,
                    body=body,
                    from_email=from_email,
                )
            except Exception as exc:
                logger.exception("SMTP delivery failed to %s", to)
                failures.append(f"SMTP delivery failed: {exc}")
        else:
            failures.append("SMTP_HOST is not configured")

        failure_message = "; ".join(failures)
        logger.error("Email was not sent to %s: %s", to, failure_message)
        raise EmailDeliveryError(f"Email was not sent: {failure_message}")

    # Reason: Helper to send emails via the official Gmail API using OAuth credentials.
    def _send_via_gmail_api(
        self,
        creds_row: Any,
        to: str,
        subject: str,
        body: str,
        from_email: str,
    ) -> dict[str, Any]:
        """Send email using Google API client with user OAuth credentials."""
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        creds = Credentials(
            token=creds_row.access_token,
            refresh_token=creds_row.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
        )

        message = MIMEText(body, "plain", "utf-8")
        message["to"] = to
        message["from"] = from_email
        message["subject"] = subject

        raw_payload = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        sent_response = service.users().messages().send(userId="me", body={"raw": raw_payload}).execute()

        msg_id = sent_response.get("id", str(uuid.uuid4()))
        record = {
            "status": "sent",
            "provider": "gmail_api",
            "message_id": msg_id,
            "to": to,
            "from": from_email,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_emails.append(record)
        logger.info("Email sent via Gmail API to %s (id=%s)", to, msg_id)
        return record

    # Reason: Helper to send emails via standard SMTP protocol.
    def _send_via_smtp(
        self,
        to: str,
        subject: str,
        body: str,
        from_email: str,
    ) -> dict[str, Any]:
        """Send email using smtplib to a configured SMTP host."""
        msg = email.message.EmailMessage()
        msg["From"] = from_email
        msg["To"] = to
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=15) as server:
            if settings.SMTP_USE_TLS:
                server.starttls()
            if settings.SMTP_USER and settings.SMTP_PASSWORD.get_secret_value():
                server.login(settings.SMTP_USER, settings.SMTP_PASSWORD.get_secret_value())
            server.send_message(msg)

        msg_id = f"smtp_{uuid.uuid4().hex[:12]}"
        record = {
            "status": "sent",
            "provider": "smtp",
            "message_id": msg_id,
            "to": to,
            "from": from_email,
            "subject": subject,
            "body": body,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._sent_emails.append(record)
        logger.info("Email sent via SMTP to %s (id=%s)", to, msg_id)
        return record

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

"""
gmail_loader.py

Concrete `EmailLoader` implementation that connects to the Gmail API and
returns emails as validated `Email` model objects.

Data flow:
    Gmail Server  --(GmailLoader)-->  List[Email]

This module only reads and structures emails. It performs no cleaning,
chunking, embedding, storage, or retrieval — those belong to later
stages of the RAG pipeline.

Authentication
--------------
Uses OAuth tokens obtained by the auth service (stored in PostgreSQL).
The loader receives an access_token and optional refresh_token — it does
NOT run its own OAuth consent flow. If the access token is expired, it
refreshes using the client credentials from config.py.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Dict, List, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from config import settings
from models import Attachment, Email
from .base_email_loader import EmailLoader, EmailLoaderAuthRevokedError, EmailLoaderConnectionError, EmailLoaderFetchError

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = settings.google_scopes


class GmailLoader(EmailLoader):
    """
    Loads emails from a Gmail account via the Gmail API.

    Parameters
    ----------
    access_token:
        OAuth access token from the auth service.
    refresh_token:
        OAuth refresh token for renewing expired access tokens.
    max_emails:
        Maximum number of emails `fetch_emails()` will return.
    label:
        Gmail label or folder to read from (e.g. "INBOX", "SENT",
        a custom label name, or a label ID).
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_expiry: Optional[datetime] = None,
        max_emails: Optional[int] = 50,
        label: str = "INBOX",
        after_timestamp: Optional[datetime] = None,
        before_timestamp: Optional[datetime] = None,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expiry = token_expiry
        self.max_emails = max_emails
        self.label = label
        self.after_timestamp = after_timestamp
        self.before_timestamp = before_timestamp

        self._service: Optional[Resource] = None
        self._creds: Optional[Credentials] = None

    @property
    def is_connected(self) -> bool:
        """Check if the loader is currently connected to the Gmail API."""
        return self._service is not None

    # ------------------------------------------------------------------
    # EmailLoader interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build credentials from stored tokens and connect to Gmail API."""
        logger.info("Connecting to Gmail API...")
        try:
            creds = self._build_credentials()
            self._creds = creds
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            logger.info("Gmail API connection established.")
        except EmailLoaderConnectionError:
            raise
        except Exception as exc:
            raise EmailLoaderConnectionError(f"Failed to connect to Gmail: {exc}") from exc

    def disconnect(self) -> None:
        """Release the Gmail API service and cached credentials."""
        if self._service is not None:
            try:
                self._service.close()
            except Exception as exc:
                logger.debug("Error while closing Gmail service: %s", exc)
        # Capture any token that google_auth_httplib2 refreshed mid-request
        if self._creds is not None and self._creds.token:
            self.access_token = self._creds.token
            self.token_expiry = self._creds.expiry
        self._service = None
        self._creds = None
        logger.info("Disconnected from Gmail API.")

    def fetch_emails(self) -> List[Email]:
        """
        Fetch up to `max_emails` emails from `label` and return them as
        validated `Email` objects. Individual messages that fail to
        download or parse are logged and skipped rather than aborting
        the whole batch.
        """
        if not self.is_connected:
            raise EmailLoaderConnectionError("Not connected. Call connect() first.")

        emails: List[Email] = []
        message_ids = self._get_message_ids()

        for message_id in message_ids:
            try:
                raw_message = self._download_message(message_id)
                email_obj = self._parse_email(message_id, raw_message)
                emails.append(email_obj)
            except Exception as exc:
                logger.warning("Skipping message %s due to error: %s", message_id, exc)

        logger.info(
            "Fetched %d/%d emails from Gmail (label=%s).",
            len(emails),
            len(message_ids),
            self.label,
        )
        return emails

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def get_current_history_id(self) -> str:
        """Return the current mailbox historyId from users.getProfile()."""
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API.")
        profile = self._service.users().getProfile(userId="me").execute()
        return profile["historyId"]

    def get_account_email(self) -> str:
        """Return the email address of the authenticated Gmail account."""
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API.")
        profile = self._service.users().getProfile(userId="me").execute()
        address = profile.get("emailAddress")
        if not address:
            raise EmailLoaderFetchError("Gmail profile did not include an email address.")
        return address

    def fetch_thread(self, thread_id: str) -> List[Email]:
        """Fetch and parse every message currently present in a Gmail thread."""
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API.")
        try:
            response = (
                self._service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to fetch Gmail thread {thread_id}: {exc}") from exc

        messages = response.get("messages", [])
        return [
            self._parse_email(message["id"], message)
            for message in messages
            if message.get("id")
        ]

    def fetch_new_by_history(self, history_id: str) -> tuple[List[Email], str]:
        """Return (new_emails, new_history_id) for messages added since history_id.

        Raises EmailLoaderFetchError if historyId has expired (>30 days gap).
        """
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API.")
        history_records = []
        page_token = None
        new_history_id = history_id
        try:
            while True:
                list_kwargs: Dict[str, Any] = {
                    "userId": "me",
                    "startHistoryId": history_id,
                    "historyTypes": ["messageAdded"],
                }
                if page_token:
                    list_kwargs["pageToken"] = page_token
                response = self._service.users().history().list(**list_kwargs).execute()
                history_records.extend(response.get("history", []))
                new_history_id = response.get("historyId", new_history_id)
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            raise EmailLoaderFetchError(f"history.list failed: {exc}") from exc

        # Gmail history can repeat a message across records; process each message once per batch.
        # (auto responder added part by saif)
        message_ids = list(dict.fromkeys(
            msg["message"]["id"]
            for record in history_records
            for msg in record.get("messagesAdded", [])
        ))
        emails: List[Email] = []
        for msg_id in message_ids:
            try:
                emails.append(self._parse_email(msg_id, self._download_message(msg_id)))
            except Exception as exc:
                logger.warning("Skipping message %s: %s", msg_id, exc)
        return emails, new_history_id

    def _build_credentials(self) -> Credentials:
        """Build Google credentials from stored tokens, refreshing if needed."""
        # google-auth expects naive UTC datetimes for credential expiry.
        expiry = self.token_expiry
        if expiry is not None and expiry.tzinfo is not None:
            expiry = expiry.astimezone(timezone.utc).replace(tzinfo=None)
        creds = Credentials(
            token=self.access_token,
            refresh_token=self.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
            scopes=DEFAULT_SCOPES,
            expiry=expiry,
        )

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Gmail token...")
                try:
                    creds.refresh(Request())
                except RefreshError as exc:
                    if "invalid_grant" in str(exc).lower():
                        raise EmailLoaderAuthRevokedError(
                            "Google refresh token has been revoked (invalid_grant)."
                        ) from exc
                    raise EmailLoaderConnectionError(f"Token refresh failed: {exc}") from exc
                self.access_token = creds.token
                self.token_expiry = creds.expiry
            else:
                raise EmailLoaderConnectionError(
                    "Access token is invalid and no refresh token is available."
                )

        return creds

    @staticmethod
    def _to_timestamp_int(val: Any) -> int:
        """Safely convert datetime, str (ISO/digits), or numeric timestamp to integer timestamp."""
        if isinstance(val, (int, float)):
            return int(val)
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return int(val.timestamp())
        if isinstance(val, str):
            stripped = val.strip()
            if stripped.isdigit():
                return int(stripped)
            parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        raise ValueError(f"Cannot parse timestamp from {type(val)}: {val}")

    def _get_message_ids(self) -> List[str]:
        """Retrieve up to `max_emails` message IDs for `label`, paginating as needed."""
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API. Use the context manager.")
        if self.max_emails is not None and self.max_emails <= 0:
            return []
        message_ids: List[str] = []
        page_token: Optional[str] = None

        parts = []
        if self.after_timestamp:
            parts.append(f"after:{self._to_timestamp_int(self.after_timestamp)}")
        if self.before_timestamp:
            parts.append(f"before:{self._to_timestamp_int(self.before_timestamp)}")
        q = " ".join(parts) or None
        try:
            while True:
                batch = 500 if self.max_emails is None else min(self.max_emails - len(message_ids), 500)
                list_kwargs: Dict[str, Any] = {
                    "userId": "me",
                    "labelIds": [self.label] if self.label else None,
                    "maxResults": batch,
                    "pageToken": page_token,
                }
                if q:
                    list_kwargs["q"] = q
                response = (
                    self._service.users()
                    .messages()
                    .list(**list_kwargs)
                    .execute()
                )
                message_ids.extend(m["id"] for m in response.get("messages", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
                if self.max_emails is not None and len(message_ids) >= self.max_emails:
                    break
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to list Gmail messages: {exc}") from exc

        return message_ids[: self.max_emails]

    def _download_message(self, message_id: str) -> Dict[str, Any]:
        """Download the full raw message payload for a single message ID."""
        if self._service is None:
            raise EmailLoaderConnectionError("Not connected to Gmail API. Use the context manager.")
        try:
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to fetch message {message_id}: {exc}") from exc

    def _extract_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Flatten Gmail's header list into a lowercase-keyed dict."""
        return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract the message body, preferring text/plain over text/html."""
        plain_text = self._find_body_by_mime_type(payload, "text/plain")
        if plain_text:
            return plain_text

        html_text = self._find_body_by_mime_type(payload, "text/html")
        return html_text or ""

    def _find_body_by_mime_type(self, payload: Dict[str, Any], mime_type: str) -> Optional[str]:
        """Recursively search MIME parts for the first match of `mime_type`."""
        if payload.get("mimeType") == mime_type:
            data = payload.get("body", {}).get("data")
            if data:
                return self._decode_base64(data)

        for part in payload.get("parts", []) or []:
            result = self._find_body_by_mime_type(part, mime_type)
            if result:
                return result

        return None

    def _extract_attachments(self, payload: Dict[str, Any]) -> List[Attachment]:
        """Recursively collect attachment metadata from the MIME tree."""
        attachments: List[Attachment] = []
        self._collect_attachments(payload, attachments)
        return attachments

    def _collect_attachments(
        self, payload: Dict[str, Any], attachments: List[Attachment]
    ) -> None:
        filename = payload.get("filename")
        body = payload.get("body", {})

        if filename and body.get("attachmentId"):
            attachments.append(
                Attachment(
                    filename=filename,
                    content_type=payload.get("mimeType"),
                    size=body.get("size"),
                )
            )

        for part in payload.get("parts", []) or []:
            self._collect_attachments(part, attachments)

    def _parse_email(self, message_id: str, raw_message: Dict[str, Any]) -> Email:
        """Convert a raw Gmail API message into a validated `Email` object."""
        payload = raw_message.get("payload", {})
        headers = self._extract_headers(payload)

        subject = headers.get("subject", "")
        _, sender_addr = parseaddr(headers.get("from", ""))
        recipients = self._parse_recipients(headers.get("to", ""))
        date = self._parse_date(headers.get("date"), raw_message.get("internalDate"))
        body = self._extract_body(payload)
        attachments = self._extract_attachments(payload)

        return Email(
            id=message_id,
            subject=subject,
            sender=sender_addr,
            recipients=recipients,
            body=body,
            date=date,
            attachments=attachments,
            gmail_thread_id=raw_message.get("threadId"),
            gmail_label_ids=raw_message.get("labelIds", []),
            gmail_internal_date_ms=(
                int(raw_message["internalDate"])
                if raw_message.get("internalDate") is not None
                else None
            ),
        )

    @staticmethod
    def _parse_recipients(to_header: str) -> List[str]:
        if not to_header:
            return []
        parsed = [parseaddr(addr)[1] for addr in to_header.split(",")]
        return [addr for addr in parsed if addr]

    @staticmethod
    def _parse_date(date_header: Optional[str], internal_date_ms: Optional[str]) -> datetime:
        if date_header:
            try:
                dt = parsedate_to_datetime(date_header)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except (TypeError, ValueError):
                pass

        if internal_date_ms:
            return datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)

        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _decode_base64(data: str) -> str:
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to decode message body: %s", exc)
            return ""

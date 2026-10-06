"""Decide whether a newly synchronized Gmail message should be answered."""

from __future__ import annotations

import logging

from initializer import chunker, db, dispatcher, embedder, store
from loaders.gmail_loader import GmailLoader
from models import Email
from rag.pipeline import Pipeline
from services.email_agent_service import process_incoming_email

logger = logging.getLogger(__name__)


class GmailAutoResponder:
    """Handle Gmail replies, manual-reply detection, and reply-dependent indexing."""

    @staticmethod
    def _is_owner_message(message: Email, account_email: str) -> bool:
        # Gmail's SENT label is a second signal in case the From header is an alias.
        return (
            message.sender.casefold() == account_email
            or "SENT" in message.gmail_label_ids
        )

    @staticmethod
    def _message_time(message: Email) -> float:
        # Gmail's server timestamp is more consistent than the sender-provided Date header.
        if message.gmail_internal_date_ms is not None:
            return message.gmail_internal_date_ms / 1000
        return message.date.timestamp()

    def process_email(self, email: Email, user_id: int) -> dict:
        """Check a synced message for a manual reply, then optionally auto-reply."""
        # Resolve credentials for the app user passed by the Gmail sync job.
        with db.session_context() as session:
            credentials = db.find_google_credentials(session, user_id)
        if credentials is None:
            return {"status": "skipped", "reason": "gmail_not_linked"}

        # Thread inspection is required to tell whether the owner already responded.
        if not email.gmail_thread_id:
            logger.warning("Skipping Gmail message %s without a thread ID.", email.id)
            return {"status": "skipped", "reason": "missing_thread_id"}

        # Load the full Gmail thread and refresh persisted tokens if Google renewed them.
        google_creds = {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "token_expiry": credentials.token_expiry,
        }
        with GmailLoader(
            access_token=google_creds["access_token"],
            refresh_token=google_creds["refresh_token"],
            token_expiry=google_creds["token_expiry"],
            label=credentials.sync_label,
        ) as loader:
            account_email = loader.get_account_email().casefold()
            thread_messages = loader.fetch_thread(email.gmail_thread_id)
            refreshed_access_token = loader.access_token
            refreshed_token_expiry = loader.token_expiry

        if refreshed_access_token != google_creds["access_token"]:
            with db.session_context() as session:
                db.update_google_access_token(
                    session, user_id, refreshed_access_token, refreshed_token_expiry
                )

        # Only incoming messages belonging to this account's selected sync label can
        # trigger a response. Owner-authored messages are considered separately.
        incoming_messages = [
            message
            for message in thread_messages
            if not self._is_owner_message(message, account_email)
            and credentials.sync_label in message.gmail_label_ids
        ]
        owner_replies = [
            message
            for message in thread_messages
            if self._is_owner_message(message, account_email)
        ]

        # Match each incoming message to the first later message sent by the owner.
        manually_replied = []
        for incoming in incoming_messages:
            later_owner_reply = next(
                (
                    reply
                    for reply in sorted(owner_replies, key=self._message_time)
                    if self._message_time(reply) > self._message_time(incoming)
                ),
                None,
            )
            if later_owner_reply is not None:
                manually_replied.append((incoming, later_owner_reply))

        if manually_replied:
            # A manual response is already the answer; index the exchange, never send
            # another reply. Return early if this is the message currently being synced.
            self._index_manual_replies(user_id, manually_replied)
            if any(incoming.id == email.id for incoming, _ in manually_replied):
                return {"status": "replied_manually", "message_id": email.id}

        # Gmail history includes sent messages too. They can help reconcile an earlier
        # incoming message above, but a sent message itself must never trigger a reply.
        if self._is_owner_message(email, account_email):
            return {
                "status": "manual_reply_reconciled" if manually_replied else "sent_message_ignored",
                "message_id": email.id,
            }

        matching_incoming = next(
            (message for message in incoming_messages if message.id == email.id), None
        )
        if matching_incoming is None:
            # The history event was not an incoming message in the configured label.
            return {"status": "skipped", "reason": "outside_sync_label"}

        # Completed reply states survive process restarts and prevent duplicate replies.
        with db.session_context() as session:
            state = db.find_gmail_reply_state(session, user_id, email.id)
            if state is not None and state.status in ("replied_auto", "replied_manual"):
                return {"status": state.status, "message_id": email.id}
            autoresponder_enabled = credentials.autoresponder_enabled

        if not autoresponder_enabled:
            # Keep an unanswered message pending. A later sync can notice a manual reply.
            with db.session_context() as session:
                db.ensure_gmail_message_pending(
                    session, user_id, email.id, email.gmail_thread_id
                )
            return {"status": "pending", "reason": "autoresponder_disabled"}

        # Atomically claim this message so overlapping syncs cannot both send a reply.
        with db.session_context() as session:
            claim_status = db.claim_gmail_message_for_autoreply(
                session, user_id, email.id, email.gmail_thread_id
            )
        if claim_status != "claimed":
            return {
                "status": claim_status,
                "message_id": email.id,
            }

        # Delegate reply composition, sending, and successful-send indexing to the
        # existing email agent workflow. Mark exceptions retryable before propagating.
        try:
            result = process_incoming_email(email, user_id=user_id)
        except Exception:
            with db.session_context() as session:
                db.save_gmail_reply_state(
                    session,
                    user_id,
                    email.id,
                    email.gmail_thread_id,
                    state="failed",
                )
            raise

        # Persist the outcome; only a successful send is marked as automatically replied.
        sent = result.get("email_sent", False)
        with db.session_context() as session:
            db.save_gmail_reply_state(
                session,
                user_id,
                email.id,
                email.gmail_thread_id,
                state="replied_auto" if sent else "deferred",
                reply_message_id=(result.get("sent_details") or {}).get("message_id"),
                indexed=sent,
            )
        return {
            "status": "replied_automatically" if sent else "pending",
            "message_id": email.id,
            "email_sent": sent,
        }

    @staticmethod
    def _index_manual_replies(
        user_id: int, exchanges: list[tuple[Email, Email]]
    ) -> None:
        """Index manually answered incoming/reply pairs once for this user."""
        unindexed_exchanges = []
        # Skip exchanges with an indexed_at marker; they were already stored earlier.
        for incoming, reply in exchanges:
            with db.session_context() as session:
                state = db.find_gmail_reply_state(session, user_id, incoming.id)
                if state is not None and state.indexed_at is not None:
                    continue
            unindexed_exchanges.append((incoming, reply))

        if not unindexed_exchanges:
            return

        # The RAG pipeline cleans, chunks, embeds, and stores both sides of each exchange.
        documents = [
            message
            for exchange in unindexed_exchanges
            for message in exchange
        ]
        Pipeline(
            store=store,
            embedder=embedder,
            dispatcher=dispatcher,
            chunker=chunker,
        ).run(documents)

        # Record successful indexing and the Gmail ID of the owner's manual reply.
        for incoming, reply in unindexed_exchanges:
            with db.session_context() as session:
                db.save_gmail_reply_state(
                    session,
                    user_id,
                    incoming.id,
                    incoming.gmail_thread_id,
                    state="replied_manual",
                    reply_message_id=reply.id,
                    indexed=True,
                )


gmail_autoresponder = GmailAutoResponder()

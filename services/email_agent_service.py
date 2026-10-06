"""
services/email_agent_service.py

Reason: Orchestrates the end-to-end incoming email automation lifecycle:
  1. Parse incoming email to extract metadata (sender, subject, recipients, date) and body.
  2. Format sender, subject, and body into a structured representation for the agent.
  3. Send the structured format and shared system prompt to the agent.
  4. Require a vector-store search before the agent composes an email reply and calls
     the `send_email_reply` tool.
  5. Ingest the incoming email and, when sent, the composed reply into ChromaDB
   through the RAG pipeline.
"""

from __future__ import annotations

import email
import email.policy
import logging
import uuid
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Optional, Union

from agent.controller import run
from agent.state import AgentState
from agent.system_prompt import SYSTEM_PROMPT
from models import Email
from schemas.email_agent import IncomingEmailRequest
from services.email_service import email_service



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reason: Normalize optional recipients while keeping reply fallback behavior.
def _normalize_recipients(
    recipients: Optional[list[str]], fallback_recipient: Optional[str] = None
) -> list[str]:
    """Return valid recipients, or the supplied fallback when available."""
    if recipients:
        valid = [r.strip() for r in recipients if r and r.strip()]
        if valid:
            return valid
    return [fallback_recipient] if fallback_recipient else []


# Reason: Helper to parse raw RFC 822 / MIME text into structured fields.
# Necessary to support standard email formats from SMTP servers, webhooks, or file uploads.
def parse_raw_mime_email(raw_mime: str) -> dict[str, Any]:
    """Parse raw RFC 822 MIME message into a normalized dictionary."""
    msg = email.message_from_string(raw_mime, policy=email.policy.default)

    # Extract sender
    from_header = msg.get("From", "")
    _, sender = parseaddr(from_header)
    if not sender:
        raise ValueError("Incoming email is missing a valid From address.")

    # Extract subject
    subject = msg.get("Subject", "")

    # Extract recipients
    to_header = msg.get("To", "")
    recipients = [parseaddr(addr)[1] for addr in to_header.split(",") if parseaddr(addr)[1]]

    # Extract date
    date_header = msg.get("Date")
    date_val = datetime.now(timezone.utc)
    if date_header:
        try:
            date_val = parsedate_to_datetime(date_header)
        except Exception:
            pass

    # Extract body
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get("Content-Disposition", ""))
            if ctype == "text/plain" and "attachment" not in cdispo:
                body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                break
        if not body:
            # Fallback to HTML if plain text not found
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    body = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
        else:
            body = str(msg.get_payload() or "")

    msg_id = msg.get("Message-ID") or f"in_{uuid.uuid4().hex[:12]}"

    return {
        "id": msg_id,
        "sender": sender,
        "subject": subject,
        "recipients": recipients,
        "body": body,
        "date": date_val,
    }


# Reason: Parsing component that accepts multiple input types (raw string, dictionary,
# IncomingEmailRequest schema, or existing Email model) and extracts metadata and body
# for subsequent vector store storage and agent reasoning.
def parse_incoming_email(
    source: Union[str, dict[str, Any], IncomingEmailRequest, Email],
) -> tuple[dict[str, Any], Email]:
    """
    Parse an incoming email from raw text, dict, or request object.
    Returns:
      (metadata_dict, email_model_instance)
    """
    # Case 1: Already an Email instance
    if isinstance(source, Email):
        metadata = {
            "id": source.id,
            "sender": source.sender,
            "subject": source.subject,
            "recipients": list(source.recipients),
            "body": source.body,
            "date": source.date,
        }
        return metadata, source

    # Case 2: Raw MIME string
    if isinstance(source, str):
        parsed = parse_raw_mime_email(source)
    # Case 3: IncomingEmailRequest schema object
    elif isinstance(source, IncomingEmailRequest):
        if source.raw_email:
            parsed = parse_raw_mime_email(source.raw_email)
        else:
            parsed = {
                "id": f"in_{uuid.uuid4().hex[:12]}",
                "sender": source.sender,
                "subject": source.subject,
                "recipients": _normalize_recipients(source.recipients),
                "body": source.body,
                "date": datetime.now(timezone.utc),
            }
    # Case 4: Standard dictionary
    elif isinstance(source, dict):
        if source.get("raw_email"):
            parsed = parse_raw_mime_email(source["raw_email"])
        else:
            sender = source.get("sender") or source.get("from")
            if not sender:
                raise ValueError("Incoming email is missing a sender address.")
            _, sender_addr = parseaddr(str(sender))
            if not sender_addr:
                raise ValueError("Incoming email has an invalid sender address.")
            sender_email = sender_addr
            recipients = source.get("recipients") or source.get("to") or []
            if isinstance(recipients, str):
                recipients = [r.strip() for r in recipients.split(",") if r.strip()]

            parsed = {
                "id": str(source.get("id") or f"in_{uuid.uuid4().hex[:12]}"),
                "sender": sender_email,
                "subject": str(source.get("subject") or ""),
                "recipients": _normalize_recipients(recipients),
                "body": str(source.get("body") or ""),
                "date": source.get("date") or datetime.now(timezone.utc),
            }
    else:
        raise ValueError(f"Unsupported incoming email payload type: {type(source)}")

    # Reason: Construct validated `Email` model instance from extracted metadata and body.
    # This instance will be persisted in the vector store after reply is sent.
    email_model = Email(
        id=parsed["id"],
        subject=parsed["subject"],
        sender=parsed["sender"],
        recipients=parsed["recipients"],
        body=parsed["body"],
        date=parsed["date"] if isinstance(parsed["date"], datetime) else datetime.now(timezone.utc),
        attachments=[],
    )

    return parsed, email_model


# Reason: Format extracted sender, subject, and body into a structured representation.
# Fulfills the requirement: "From the metadata send the sender, subject and body in a structure format,
# the system prompt and the tools to the agent."
def format_email_for_agent(sender: str, subject: str, body: str) -> str:
    """Format the incoming email attributes into a structured prompt for the agent."""
    # The shared agent prompt recognizes this marker as an incoming-email request.
    # (auto responder added part by saif)
    return f"""New email
Sender: {sender}
Subject: {subject}
Body:
{body}
"""


# Reason: Primary orchestration service function implementing the full workflow:
# 1. Parse incoming email (metadata + body).
# 2. Package structured prompt, system prompt, and tools to the agent.
# 3. Agent searches the vector store, composes the email, and calls send_email_reply.
# 4. On successful sending, stores both the incoming email and reply in the ChromaDB vector store.
def process_incoming_email(
    incoming_data: Union[str, dict[str, Any], IncomingEmailRequest, Email],
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Process an incoming email with the agent, send a reply, and persist both
    in the vector store upon successful delivery.
    """
    logger.info("Starting automated incoming email processing...")

    # Step 1: Parse the incoming email to extract metadata and body
    metadata, incoming_email_obj = parse_incoming_email(incoming_data)
    logger.info("Parsed incoming email ID=%s.", incoming_email_obj.id)

    # Step 2: Format sender, subject, and body into a structured representation
    structured_user_prompt = format_email_for_agent(
        sender=incoming_email_obj.sender,
        subject=incoming_email_obj.subject,
        body=incoming_email_obj.body,
    )

    # Step 4: Run the shared knowledge agent in incoming-email mode.
    # (auto responder added part by saif)
    state = AgentState(user_input=structured_user_prompt)
    email_service.clear_current_sent_email()

    agent_final_text = run(
        state,
        system_prompt=SYSTEM_PROMPT,
        user_id=user_id,
        allow_email_reply=True,
        # The tool obtains the destination from private context, not its schema.
        email_reply_to=incoming_email_obj.sender,
    )
    logger.info("Agent processing complete for incoming email %s.", incoming_email_obj.id)

    # Step 5: Check whether the reply tool successfully dispatched an email.
    last_sent = email_service.get_current_sent_email()
    email_was_sent = last_sent is not None

    chunks_stored = 0
    reply_email_obj = None

    # Step 6: After successful sending, store both messages in the vector store.
    if email_was_sent:
        logger.info("Detected successful email dispatch. Preparing both emails for vector store ingestion.")

        # Determine reply content from sent email record or tool calls
        reply_subject = last_sent.get("subject") or f"Re: {incoming_email_obj.subject}"
        reply_to = last_sent.get("to") or incoming_email_obj.sender
        reply_body = last_sent.get("body") or agent_final_text
        reply_from = last_sent["from"]

        # Reason: Construct validated `Email` model for the generated reply email.
        reply_email_obj = Email(
            id=f"reply_{uuid.uuid4().hex[:12]}",
            subject=reply_subject,
            sender=reply_from,
            recipients=_normalize_recipients([reply_to], incoming_email_obj.sender),
            body=reply_body,
            date=datetime.now(timezone.utc),
            attachments=[],
        )

        # Reason: Store both incoming email and composed reply email in ChromaDB vector store
        # using the existing unified RAG pipeline (cleaning -> chunking -> embedding -> storage).
        try:
            from initializer import chunker, dispatcher, embedder, store
            from rag.pipeline import Pipeline

            rag_pipeline = Pipeline(
                store=store,
                embedder=embedder,
                dispatcher=dispatcher,
                chunker=chunker,
            )
            chunks_stored = rag_pipeline.run([incoming_email_obj, reply_email_obj])
            logger.info(
                "Ingested %d chunks into vector store for incoming (%s) and reply (%s)",
                chunks_stored,
                incoming_email_obj.id,
                reply_email_obj.id,
            )
        except Exception as exc:
            logging.exception("Failed to store emails in vector store: %s", exc)

        return {
            "status": "success",
            "message": "Email processed, reply sent, and conversation stored in vector store.",
            "email_sent": True,
            "incoming_email_id": incoming_email_obj.id,
            "sender": incoming_email_obj.sender,
            "subject": incoming_email_obj.subject,
            "sent_details": last_sent,
            "reply_email_id": reply_email_obj.id if reply_email_obj else None,
            "chunks_stored": chunks_stored,
            "agent_reply": agent_final_text,
        }

    # Never index a new email until its reply has been successfully sent.
    # (auto responder added part by saif)
    logger.warning("No reply was sent for email %s; it was not indexed.", incoming_email_obj.id)
    return {
        "status": "no_email_sent",
        "message": "Agent finished without sending a reply. The incoming email was not indexed.",
        "email_sent": False,
        "incoming_email_id": incoming_email_obj.id,
        "sender": incoming_email_obj.sender,
        "subject": incoming_email_obj.subject,
        "sent_details": None,
        "reply_email_id": None,
        "chunks_stored": chunks_stored,
        "agent_reply": agent_final_text,
    }

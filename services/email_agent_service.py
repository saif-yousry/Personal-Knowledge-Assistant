"""
services/email_agent_service.py

Reason: Orchestrates the end-to-end incoming email automation lifecycle:
  1. Parse incoming email to extract metadata (sender, subject, recipients, date) and body.
  2. Format sender, subject, and body into a structured representation for the agent.
  3. Send the structured format, system prompt, and tools (search and send_email) to the agent.
  4. Allow the agent to reason, optionally retrieve context via vector store `search`,
     compose an email reply, and call the `send_email` tool.
  5. Upon successful dispatch, ingest both the incoming email and the composed reply
     into the ChromaDB vector store via the RAG pipeline.
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
from agent.system_prompt import EMAIL_AGENT_SYSTEM_PROMPT
from config import settings
from models import Email
from schemas.email_agent import IncomingEmailRequest
from services.email_service import email_service



logging.basicConfig(level=logging.INFO)

# Reason: Helper to safely extract or generate email recipients to guarantee
# that the `Email` Pydantic model's `at_least_one_recipient` validator is satisfied.
def _normalize_recipients(recipients: Optional[list[str]], fallback_recipient: str) -> list[str]:
    """Ensure at least one valid recipient is present for Email model validation."""
    if recipients:
        valid = [r.strip() for r in recipients if r and r.strip()]
        if valid:
            return valid
    return [fallback_recipient]


# Reason: Helper to parse raw RFC 822 / MIME text into structured fields.
# Necessary to support standard email formats from SMTP servers, webhooks, or file uploads.
def parse_raw_mime_email(raw_mime: str) -> dict[str, Any]:
    """Parse raw RFC 822 MIME message into a normalized dictionary."""
    msg = email.message_from_string(raw_mime, policy=email.policy.default)

    # Extract sender
    from_header = msg.get("From", "")
    _, sender = parseaddr(from_header)
    if not sender:
        sender = settings.DEFAULT_SENDER_EMAIL

    # Extract subject
    subject = msg.get("Subject", "")

    # Extract recipients
    to_header = msg.get("To", "")
    recipients = [parseaddr(addr)[1] for addr in to_header.split(",") if parseaddr(addr)[1]]
    if not recipients:
        recipients = [settings.DEFAULT_SENDER_EMAIL]

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
                "recipients": _normalize_recipients(source.recipients, settings.DEFAULT_SENDER_EMAIL),
                "body": source.body,
                "date": datetime.now(timezone.utc),
            }
    # Case 4: Standard dictionary
    elif isinstance(source, dict):
        if source.get("raw_email"):
            parsed = parse_raw_mime_email(source["raw_email"])
        else:
            sender = source.get("sender") or source.get("from") or settings.DEFAULT_SENDER_EMAIL
            _, sender_addr = parseaddr(str(sender))
            sender_email = sender_addr if sender_addr else settings.DEFAULT_SENDER_EMAIL
            recipients = source.get("recipients") or source.get("to") or [settings.DEFAULT_SENDER_EMAIL]
            if isinstance(recipients, str):
                recipients = [r.strip() for r in recipients.split(",") if r.strip()]

            parsed = {
                "id": str(source.get("id") or f"in_{uuid.uuid4().hex[:12]}"),
                "sender": sender_email,
                "subject": str(source.get("subject") or ""),
                "recipients": _normalize_recipients(recipients, settings.DEFAULT_SENDER_EMAIL),
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
    return f"""--- INCOMING EMAIL METADATA & BODY ---
Sender: {sender}
Subject: {subject}
Body:
{body}
---------------------------------------

Instructions:
1. Reason about the incoming email and the sender's inquiry or intent.
2. If specific context, facts, or past records are needed to provide an accurate reply, use the `search` tool to retrieve relevant documents from the vector store.
3. Formulate a polite, professional, and helpful email reply.
4. Call the `send_email` tool to dispatch the reply to '{sender}'.
"""


# Reason: Primary orchestration service function implementing the full workflow:
# 1. Parse incoming email (metadata + body).
# 2. Package structured prompt, system prompt, and tools to the agent.
# 3. Agent reasons, searches vector store if needed, composes email, and calls send_email tool.
# 4. On successful sending, stores both the incoming email and reply in the ChromaDB vector store.
def process_incoming_email(
    incoming_data: Union[str, dict[str, Any], IncomingEmailRequest, Email],
    user_id: Optional[int] = None,
) -> dict[str, Any]:
    """
    Process an incoming email with the agent, send a reply, and persist both
    in the vector store upon successful delivery.
    """
    logging.INFO("Starting automated incoming email processing...")

    # Step 1: Parse the incoming email to extract metadata and body
    metadata, incoming_email_obj = parse_incoming_email(incoming_data)
    logging.INFO("Parsed incoming email ID=%s from '%s'", incoming_email_obj.id, incoming_email_obj.sender)

    # Step 2: Format sender, subject, and body into a structured representation
    structured_user_prompt = format_email_for_agent(
        sender=incoming_email_obj.sender,
        subject=incoming_email_obj.subject,
        body=incoming_email_obj.body,
    )

    # Step 3: Record initial count of sent emails to verify new email dispatch
    initial_sent_count = len(email_service.get_sent_emails())

    # Step 4: Initialize AgentState and run agent loop with system prompt and tools
    state = AgentState(user_input=structured_user_prompt)

    # Run the agentic loop (tools `search` and `send_email` are automatically supplied via TOOL_REGISTRY)
    agent_final_text = run(state, system_prompt=EMAIL_AGENT_SYSTEM_PROMPT)
    logging.INFO("Agent processing complete. Final reply snippet: %s", agent_final_text[:120])

    # Step 5: Check whether `send_email` was successfully called during the agent run
    current_sent_emails = email_service.get_sent_emails()
    email_was_sent = len(current_sent_emails) > initial_sent_count
    last_sent = current_sent_emails[-1] if email_was_sent else None

    # Alternatively inspect state messages for successful send_email tool execution
    if not email_was_sent:
        for msg in state.messages:
            if msg.get("role") == "tool" and "[send_email] result:" in msg.get("content", ""):
                email_was_sent = True
                break

    chunks_stored = 0
    reply_email_obj = None

    # Step 6: "After successful sending the new email and reply are stored in the vector store."
    if email_was_sent:
        logging.INFO("Detected successful email dispatch. Preparing both emails for vector store ingestion.")

        # Determine reply content from sent email record or tool calls
        if last_sent:
            reply_subject = last_sent.get("subject") or f"Re: {incoming_email_obj.subject}"
            reply_to = last_sent.get("to") or incoming_email_obj.sender
            reply_body = last_sent.get("body") or agent_final_text
            reply_from = last_sent.get("from") or settings.DEFAULT_SENDER_EMAIL
        else:
            reply_subject = f"Re: {incoming_email_obj.subject}"
            reply_to = incoming_email_obj.sender
            reply_body = agent_final_text
            reply_from = settings.DEFAULT_SENDER_EMAIL

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
            logging.INFO(
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

    logging.warning("No email was sent by the agent. Emails will not be persisted in vector store.")
    return {
        "status": "no_email_sent",
        "message": "Agent finished without sending an email. Vector store storage skipped.",
        "email_sent": False,
        "incoming_email_id": incoming_email_obj.id,
        "sender": incoming_email_obj.sender,
        "subject": incoming_email_obj.subject,
        "sent_details": None,
        "reply_email_id": None,
        "chunks_stored": 0,
        "agent_reply": agent_final_text,
    }

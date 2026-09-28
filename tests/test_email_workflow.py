"""
tests/test_email_workflow.py

Tests for the email sending tool and automated incoming email processing workflow:
1. Parsing incoming emails into metadata, body, and validated Email models.
2. Structured prompt formatting for the agent.
3. Execution of the `send_email` tool and schema validation.
4. End-to-end agentic reasoning with mocked LLM tool calls.
5. Verification that incoming email and reply are stored in the vector store after successful dispatch.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from agent.tool_executer import TOOL_REGISTRY, execute_tool
from agent.llm_client import _build_tools
from agent.tools import send_email
from models import Email
from schemas.email_agent import IncomingEmailRequest
from services.email_agent_service import (
    format_email_for_agent,
    parse_incoming_email,
    process_incoming_email,
)
from services.email_service import email_service


class TestEmailAgentWorkflow(unittest.TestCase):

    def setUp(self):
        email_service.clear_history()

    def test_send_email_registered_in_tool_registry(self):
        """Verify send_email is registered in TOOL_REGISTRY with correct schema."""
        self.assertIn("send_email", TOOL_REGISTRY)
        tools = _build_tools()
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("send_email", tool_names)
        self.assertIn("search", tool_names)

    def test_send_email_tool_execution(self):
        """Verify execute_tool invokes send_email and returns a successful ToolResult."""
        args = {
            "to": "test_client@example.com",
            "subject": "Status Update",
            "body": "Your project is on schedule.",
        }
        result = execute_tool("send_email", args)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "send_email")
        self.assertIn("test_client@example.com", result.to_observation_text())

        # Verify email was recorded in email_service
        last_sent = email_service.get_last_sent_email()
        self.assertIsNotNone(last_sent)
        self.assertEqual(last_sent["to"], "test_client@example.com")
        self.assertEqual(last_sent["subject"], "Status Update")

    def test_send_email_tool_missing_arguments(self):
        """Verify send_email fails gracefully if required parameters are missing."""
        result = execute_tool("send_email", {"subject": "No recipient", "body": "Hello"})
        self.assertFalse(result.success)
        self.assertIn("to", result.error)

    def test_parse_incoming_email_from_dict(self):
        """Verify parsing a dictionary payload extracts metadata and body into an Email model."""
        payload = {
            "sender": "sender@example.com",
            "subject": "Question about pricing",
            "body": "Can you provide pricing details for enterprise tier?",
            "recipients": ["support@example.com"],
        }
        metadata, email_model = parse_incoming_email(payload)
        self.assertEqual(metadata["sender"], "sender@example.com")
        self.assertEqual(metadata["subject"], "Question about pricing")
        self.assertEqual(metadata["body"], payload["body"])
        self.assertIsInstance(email_model, Email)
        self.assertEqual(email_model.sender, "sender@example.com")
        self.assertEqual(len(email_model.recipients), 1)

    def test_parse_incoming_email_from_raw_mime(self):
        """Verify parsing raw MIME text extracts headers and body correctly."""
        raw_mime = (
            "From: alice@example.com\r\n"
            "To: bob@example.com\r\n"
            "Subject: Important Meeting\r\n"
            "Date: Mon, 28 Sep 2026 10:00:00 +0000\r\n"
            "\r\n"
            "Hello Bob, please review the Q3 budget."
        )
        metadata, email_model = parse_incoming_email(raw_mime)
        self.assertEqual(metadata["sender"], "alice@example.com")
        self.assertEqual(metadata["subject"], "Important Meeting")
        self.assertIn("review the Q3 budget", metadata["body"])
        self.assertIsInstance(email_model, Email)

    def test_format_email_for_agent(self):
        """Verify structured format includes sender, subject, and body."""
        structured = format_email_for_agent(
            sender="carol@example.com",
            subject="Project Sync",
            body="When is our next sync meeting?",
        )
        self.assertIn("carol@example.com", structured)
        self.assertIn("Project Sync", structured)
        self.assertIn("When is our next sync meeting?", structured)
        self.assertIn("search", structured)
        self.assertIn("send_email", structured)

    @patch("agent.controller.call_llm")
    @patch("rag.pipeline.Pipeline.run")
    def test_process_incoming_email_full_flow(self, mock_pipeline_run, mock_call_llm):
        """
        Verify the complete workflow:
        1. Parse email.
        2. LLM reasons and decides to call send_email.
        3. Email is dispatched.
        4. Both incoming email and reply are stored in vector store via Pipeline.run.
        """
        mock_pipeline_run.return_value = 2  # Simulating 2 chunks stored

        # Mock LLM turn 1: Call send_email
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_abc123"
        mock_tool_call.function.name = "send_email"
        mock_tool_call.function.arguments = (
            '{"to": "david@example.com", "subject": "Re: Inquiry", "body": "Thank you for reaching out!"}'
        )

        mock_msg_turn1 = MagicMock()
        mock_msg_turn1.content = None
        mock_msg_turn1.tool_calls = [mock_tool_call]

        # Mock LLM turn 2: Final response confirming email was sent
        mock_msg_turn2 = MagicMock()
        mock_msg_turn2.content = "I have replied to David with the requested information."
        mock_msg_turn2.tool_calls = []

        mock_call_llm.side_effect = [mock_msg_turn1, mock_msg_turn2]

        request_payload = IncomingEmailRequest(
            sender="david@example.com",
            subject="Inquiry",
            body="Do you have availability next Tuesday?",
        )

        result = process_incoming_email(request_payload)

        # Assertions
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["email_sent"])
        self.assertEqual(result["sender"], "david@example.com")
        self.assertIsNotNone(result["reply_email_id"])
        self.assertEqual(result["chunks_stored"], 2)

        # Verify pipeline.run was called with 2 emails: incoming and reply
        mock_pipeline_run.assert_called_once()
        documents_passed = mock_pipeline_run.call_args[0][0]
        self.assertEqual(len(documents_passed), 2)
        incoming_doc, reply_doc = documents_passed
        self.assertEqual(incoming_doc.sender, "david@example.com")
        self.assertEqual(reply_doc.recipients, ["david@example.com"])
        self.assertEqual(reply_doc.subject, "Re: Inquiry")


if __name__ == "__main__":
    unittest.main()

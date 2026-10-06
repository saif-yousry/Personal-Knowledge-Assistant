"""
email_cleaner.py

Concrete `Cleaner` implementation for `Email` objects.

Data flow:
    List[Email]  --(EmailCleaner)-->  List[Email]   (body cleaned, all
                                                       other fields kept)

Only cleans email body content — HTML markup, signatures, quoted reply
history, and whitespace. No chunking, embedding, storage, or retrieval;
those are later RAG pipeline stages.
"""

from __future__ import annotations
import re
from typing import List, Optional, Pattern

from models import Email

from .base_cleaner import Cleaner
from .cleaner_utils import html_to_text, looks_like_html, normalize_whitespace, strip_empty_lines


class EmailCleaner(Cleaner):
    """
    Cleans the body of an `Email`.

    All heuristics here are email-specific (unlike utils.py, which stays
    generic), so a future `SlackMessageCleaner` or `PdfCleaner` would
    define its own signature/quote patterns while still implementing the
    same `Cleaner` contract.
    """

    # Lines that mark the start of a quoted/forwarded-reply block. The
    # earliest match of any of these truncates the rest of the body.
    _REPLY_HISTORY_PATTERNS: List[Pattern[str]] = [
        # "On Mon, Jul 20, 2026 at 9:00 AM, John Doe <john@example.com> wrote:"
        re.compile(r"^\s*On .{0,200}?wrote:\s*$", re.IGNORECASE | re.MULTILINE),
        # Outlook-style "-----Original Message-----"
        re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.IGNORECASE | re.MULTILINE),
        # Outlook-style forwarded header block ("From: ... Sent: ... To: ...")
        re.compile(r"^\s*From:\s*.+\n\s*Sent:\s*.+\n\s*To:\s*.+", re.IGNORECASE | re.MULTILINE),
        # Gmail/Apple Mail forwarded message markers
        re.compile(r"^\s*-{2,}\s*Forwarded message\s*-{2,}\s*$", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*Begin forwarded message:\s*$", re.IGNORECASE | re.MULTILINE),
    ]

    # A line consisting only of quote markers, e.g. "> " or ">> some text"
    _QUOTE_LINE_PATTERN = re.compile(r"^\s*>.*$", re.MULTILINE)

    # Common sign-off phrases that typically precede a signature block.
    # Matched as a *whole line* so we don't clip "Thanks for the update."
    _SIGNOFF_PATTERN = re.compile(
        r"^\s*(best regards|kind regards|warm regards|regards|"
        r"best wishes|best|sincerely|many thanks|thanks so much|"
        r"thank you|thanks|cheers)\s*[,.!]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Standard RFC 3676 signature delimiter: a line that is exactly "-- "
    _SIG_DELIMITER_PATTERN = re.compile(r"^-- ?\s*$", re.MULTILINE)

    def clean(self, source: Email) -> Email:
        """Return a new `Email` with a cleaned body; all other fields preserved."""
        body = source.body or ""

        body = self._remove_html(body)
        body = self._remove_reply_history(body)
        body = self._remove_signature(body)
        body = self._normalize_whitespace(body)
        body = self._strip_empty_lines(body)

        return source.model_copy(update={"body": body})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_html(self, body: str) -> str:
        """Strip HTML tags/scripts/styles/blockquotes and decode entities."""
        if not body:
            return ""
        if looks_like_html(body):
            return html_to_text(body)
        return body


    def _remove_reply_history(self, body: str) -> str:
        """
        Truncate the body at the earliest quoted/forwarded-reply marker,
        then drop any remaining lines that are pure '>' quote markers.
        """
        if not body:
            return ""

        earliest_index: Optional[int] = None
        for pattern in self._REPLY_HISTORY_PATTERNS:
            match = pattern.search(body)
            if match and (earliest_index is None or match.start() < earliest_index):
                earliest_index = match.start()

        if earliest_index is not None:
            body = body[:earliest_index]

        return self._QUOTE_LINE_PATTERN.sub("", body)


    def _remove_signature(self, body: str) -> str:
        """Truncate the body at the first sign-off phrase or RFC 3676 '-- ' delimiter."""
        if not body:
            return ""

        cut_index: Optional[int] = None

        sig_delim_match = self._SIG_DELIMITER_PATTERN.search(body)
        if sig_delim_match:
            cut_index = sig_delim_match.start()

        signoff_match = self._SIGNOFF_PATTERN.search(body)
        if signoff_match and (cut_index is None or signoff_match.start() < cut_index):
            cut_index = signoff_match.start()

        if cut_index is not None:
            body = body[:cut_index]

        return body


    def _normalize_whitespace(self, body: str) -> str:
        """Collapse runs of spaces/tabs and trim trailing whitespace per line."""
        return normalize_whitespace(body)


    def _strip_empty_lines(self, body: str) -> str:
        """Collapse consecutive blank lines and trim leading/trailing blank space."""
        return strip_empty_lines(body)

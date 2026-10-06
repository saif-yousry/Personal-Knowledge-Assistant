"""
pdf_cleaner.py

Cleaner for PDF `Document` objects.

Handles noise typical of PDF text extraction:
- Standalone page numbers
- Hyphenated line breaks from column layouts
- Excessive whitespace

Uses shared helpers from cleaner_utils.py for generic text normalization.
"""

from __future__ import annotations

import re

from models import Document
from .base_cleaner import Cleaner
from .cleaner_utils import normalize_whitespace, strip_empty_lines


class PdfCleaner(Cleaner):
    """Cleans the text of a `Document` extracted from a PDF."""

    # A line that is just a page number (e.g. "12", "- 3 -", "Page 5")
    _PAGE_NUMBER_PATTERN = re.compile(
        r"^\s*(?:-\s*)?\d+(?:\s*-\s*)?$|^\s*page\s+\d+\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Hyphenated line break: word fragment ending with "-" followed by a newline
    _HYPHENATED_BREAK_PATTERN = re.compile(r"(\w)-\n(\w)")

    def clean(self, source: Document) -> Document:
        """Return a new `Document` with cleaned text; all other fields preserved."""
        text = source.text or ""

        text = self._remove_page_numbers(text)
        text = self._fix_hyphenated_breaks(text)
        text = normalize_whitespace(text)
        text = strip_empty_lines(text)

        return source.model_copy(update={"text": text})

    def _remove_page_numbers(self, text: str) -> str:
        """Remove lines that are just page numbers."""
        return self._PAGE_NUMBER_PATTERN.sub("", text)

    def _fix_hyphenated_breaks(self, text: str) -> str:
        """Rejoin words split across lines by a hyphen."""
        return self._HYPHENATED_BREAK_PATTERN.sub(r"\1\2", text)

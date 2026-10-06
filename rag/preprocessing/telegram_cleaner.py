"""
telegram_cleaner.py

Concrete ``Cleaner`` implementation for ``TelegramMessage`` objects.

Cleaning pipeline:
  1. Normalize line endings.
  2. Preserve fenced code blocks as-is.
  3. Strip URLs (configurable).
  4. Convert Unicode emoji to text tags.
  5. Expand common abbreviations.
  6. Normalize whitespace.
  7. Deduplicate (batch operation).
"""

from __future__ import annotations

import re
from typing import List

from models import TelegramMessage
from .base_cleaner import Cleaner
from .cleaner_utils import convert_unicode_emoji, deduplicate_messages, expand_abbreviations


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

FENCED_CODE_BLOCK = re.compile(r"(```[\s\S]*?```)")
URL = re.compile(r"(https?://[^\s]+)")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#\w+")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _clean_non_url_text(text: str, strip_urls: bool = True) -> str:
    if strip_urls:
        text = URL.sub("", text)
    text = convert_unicode_emoji(text)
    return expand_abbreviations(text)


def _clean_plain_text(text: str, strip_urls: bool = True) -> str:
    text = _clean_non_url_text(text, strip_urls=strip_urls)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _clean_content(content: str, strip_urls: bool = True) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    parts = FENCED_CODE_BLOCK.split(normalized)
    cleaned_parts = [
        part if index % 2 else _clean_plain_text(part, strip_urls=strip_urls)
        for index, part in enumerate(parts)
    ]
    return "".join(cleaned_parts).strip()


# ---------------------------------------------------------------------------
# TelegramCleaner — implements the Cleaner contract
# ---------------------------------------------------------------------------

class TelegramCleaner(Cleaner):
    """
    Cleans the text of a ``TelegramMessage``.

    Handles Telegram-specific noise: emoji, abbreviation expansion,
    code block preservation, URL stripping, and whitespace normalization.
    """

    def __init__(self, strip_urls: bool = True):
        self.strip_urls = strip_urls

    def clean(self, source: TelegramMessage) -> TelegramMessage:
        """Return a new ``TelegramMessage`` with cleaned text; all other fields preserved."""
        raw_text = source.text or ""

        cleaned = _clean_content(raw_text, strip_urls=self.strip_urls)

        if not cleaned:
            return source.model_copy(update={"text": ""})

        return source.model_copy(update={"text": cleaned})


__all__ = ["TelegramCleaner", "deduplicate_messages"]

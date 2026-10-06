"""
discord_cleaner.py

Concrete ``Cleaner`` implementation for ``DiscordMessage`` objects.

Cleaning pipeline:
  1. Normalize line endings.
  2. Preserve fenced code blocks as-is.
  3. Remove custom emoji markup, convert Unicode emoji to text.
  4. Replace user/role/channel mentions with readable labels.
  5. Expand common abbreviations (lol, btw, etc.).
  6. Normalize whitespace.
  7. Append sticker names if present in metadata.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Dict, List, Optional

from models import DiscordMessage
from .base_cleaner import Cleaner
from .cleaner_utils import convert_unicode_emoji, deduplicate_messages, expand_abbreviations


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

FENCED_CODE_BLOCK = re.compile(r"(```[\s\S]*?```)")
CUSTOM_EMOJI = re.compile(r"<a?:([A-Za-z0-9_~]+):\d+>")
USER_MENTION = re.compile(r"<@!?\d+>")
ROLE_MENTION = re.compile(r"<@&\d+>")
CHANNEL_MENTION = re.compile(r"<#\d+>")
URL = re.compile(r"(https?://[^\s]+)")


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def _readable_custom_emoji(match: re.Match[str]) -> str:
    return match.group(1).replace("_", " ")


def _clean_non_url_text(text: str) -> str:
    text = CUSTOM_EMOJI.sub(_readable_custom_emoji, text)
    text = USER_MENTION.sub("@user", text)
    text = ROLE_MENTION.sub("@role", text)
    text = CHANNEL_MENTION.sub("#channel", text)
    text = convert_unicode_emoji(text)
    return expand_abbreviations(text)


def _replace_known_mentions(text: str, mention_labels: Mapping[str, str]) -> str:
    for markup, label in mention_labels.items():
        text = text.replace(markup, label)
    return text


def _clean_plain_text(text: str, mention_labels: Mapping[str, str]) -> str:
    text = "".join(
        part
        if index % 2
        else _clean_non_url_text(_replace_known_mentions(part, mention_labels))
        for index, part in enumerate(URL.split(text))
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _clean_content(
    content: str,
    mention_labels: Mapping[str, str] | None = None,
    sticker_names: Sequence[str] | None = None,
) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    labels = mention_labels or {}
    parts = FENCED_CODE_BLOCK.split(normalized)
    cleaned_parts = [
        part if index % 2 else _clean_plain_text(part, labels)
        for index, part in enumerate(parts)
    ]
    cleaned = "".join(cleaned_parts).strip()
    stickers = [
        f"[sticker: {name.strip()}]"
        for name in (sticker_names or [])
        if isinstance(name, str) and name.strip()
    ]
    return " ".join([part for part in (cleaned, *stickers) if part])


# ---------------------------------------------------------------------------
# DiscordCleaner — implements the Cleaner contract
# ---------------------------------------------------------------------------

class DiscordCleaner(Cleaner):
    """
    Cleans the text of a ``DiscordMessage``.

    Handles Discord-specific noise: custom emoji, user/role/channel mentions,
    abbreviation expansion, Unicode emoji demojification, code block preservation,
    and whitespace normalization.
    """

    def __init__(
        self,
        mention_labels: Optional[Dict[str, str]] = None,
    ):
        self.mention_labels = mention_labels or {}

    def clean(self, source: DiscordMessage) -> DiscordMessage:
        """Return a new ``DiscordMessage`` with cleaned text; all other fields preserved."""
        raw_text = source.text or ""

        sticker_names = source.metadata.get("sticker_names") if source.metadata else None

        cleaned = _clean_content(
            raw_text,
            mention_labels=self.mention_labels,
            sticker_names=sticker_names,
        )

        if not cleaned:
            return source.model_copy(update={"text": ""})

        return source.model_copy(update={"text": cleaned})


__all__ = ["DiscordCleaner", "deduplicate_messages"]

"""
slack_cleaner.py

Concrete ``Cleaner`` implementation for ``SlackMessage`` objects.

Data flow:
    SlackMessage  --(SlackCleaner)-->  SlackMessage   (text cleaned, all
                                                       other fields kept)

Cleaning pipeline (in order):
  1. Boilerplate detection — flag system messages (join/leave/topic) as empty.
  2. Technical span protection — mask code blocks, inline code, file paths,
     CUDA errors, GPU names, env vars, IPs so later steps can't corrupt them.
  3. Slack structural resolution — mentions, channel refs, links, markdown,
     emoji shortcodes converted to plain readable text.
  4. Noise reduction — elongation normalization, whitespace collapse.
  5. Technical span restoration — put masked spans back verbatim.
  6. Deduplication — exact (SHA-256) and near-duplicate (difflib) detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from models import SlackMessage
from .base_cleaner import Cleaner
from .cleaner_utils import convert_unicode_emoji, deduplicate_messages, expand_abbreviations, hash_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlackCleaningConfig:
    protect_technical_spans: bool = True
    resolve_mentions: bool = True
    resolve_channel_refs: bool = True
    resolve_links: bool = True
    strip_markdown_formatting: bool = True
    convert_emoji_shortcodes: bool = True
    convert_unicode_emoji: bool = True
    expand_abbreviations: bool = True
    normalize_whitespace: bool = True
    normalize_elongation: bool = True
    drop_boilerplate: bool = True
    max_repeated_chars: int = 2
    near_dup_similarity_threshold: float = 0.92


# ---------------------------------------------------------------------------
# Technical span protection
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_TECHNICAL_HEURISTIC_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("job_id", re.compile(r"\b(?:job[_\s-]?id\s*[:#]?\s*|job\s+)(\d{4,})\b", re.IGNORECASE)),
    ("slurm_cmd", re.compile(r"\b(?:sbatch|squeue|srun|scancel|sinfo|sacct|salloc|scontrol)\b(?:\s+\S+)*")),
    ("gpu_name", re.compile(r"\b(?:[AVH]\d{2,3}|RTX\s?\d{3,4}(?:\s?Ti)?|GTX\s?\d{3,4}(?:\s?Ti)?|Tesla\s?[A-Z]\d+|MI\d{3})\b")),
    ("cuda_error", re.compile(r"\bCUDA_[A-Z_]+\b|\bcuda[A-Za-z]*Error[A-Za-z]*\b")),
    ("file_path", re.compile(r"(?:/[\w.\-]+){2,}/?|[A-Za-z]:\\(?:[\w.\- ]+\\)+[\w.\- ]*")),
    ("ip_address", re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")),
    ("env_var", re.compile(r"\b[A-Z][A-Z0-9_]{2,}=\S+")),
]

_PLACEHOLDER_TEMPLATE = "\u0001PROTECTED_{i}\u0001"
_PLACEHOLDER_RE = re.compile(r"\u0001PROTECTED_(\d+)\u0001")


def _protect_technical_spans(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []

    def _mask(match: re.Match) -> str:
        spans.append(match.group(0))
        return _PLACEHOLDER_TEMPLATE.format(i=len(spans) - 1)

    text = _CODE_BLOCK_RE.sub(_mask, text)
    text = _INLINE_CODE_RE.sub(_mask, text)
    for _name, pattern in _TECHNICAL_HEURISTIC_PATTERNS:
        text = pattern.sub(_mask, text)
    return text, spans


def _restore_technical_spans(text: str, spans: List[str]) -> str:
    def _unmask(match: re.Match) -> str:
        idx = int(match.group(1))
        return spans[idx]
    return _PLACEHOLDER_RE.sub(_unmask, text)


# ---------------------------------------------------------------------------
# Slack structural cleaning
# ---------------------------------------------------------------------------

_MENTION_RE = re.compile(r"<@([UW][A-Z0-9]+)(?:\|([^>]+))?>")
_CHANNEL_REF_RE = re.compile(r"<#(C[A-Z0-9]+)(?:\|([^>]+))?>")
_SPECIAL_MENTION_RE = re.compile(r"<!(here|channel|everyone)>")
_LINK_RE = re.compile(r"<(https?://[^|>\s]+)(?:\|([^>]+))?>")
_BOLD_RE = re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)")
_ITALIC_RE = re.compile(r"(?<!\w)_([^_\n]+)_(?!\w)")
_STRIKE_RE = re.compile(r"(?<!\w)~([^~\n]+)~(?!\w)")
_EMOJI_SHORTCODE_RE = re.compile(r":([a-zA-Z0-9_+\-]+):")

_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*<[^>]+>\s+has joined the channel\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*<[^>]+>\s+has left the channel\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*<[^>]+>\s+set the channel (topic|purpose)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*<[^>]+>\s+(pinned|unpinned) (a|the) message.*$", re.IGNORECASE),
    re.compile(r"^\s*<[^>]+>\s+renamed the channel.*$", re.IGNORECASE),
    re.compile(r"^\s*<[^>]+>\s+added an integration.*$", re.IGNORECASE),
]

_BOILERPLATE_SUBTYPES = {
    "channel_join", "channel_leave", "channel_topic", "channel_purpose",
    "channel_name", "channel_archive", "channel_unarchive", "pinned_item",
    "unpinned_item",
}


def _is_boilerplate(text: str, metadata: Dict[str, Any]) -> bool:
    subtype = (metadata or {}).get("subtype")
    if subtype in _BOILERPLATE_SUBTYPES:
        return True
    stripped = text.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in _BOILERPLATE_PATTERNS)


def _resolve_mentions(text: str, user_id_map: Optional[Dict[str, str]]) -> str:
    user_id_map = user_id_map or {}

    def _sub(match: re.Match) -> str:
        user_id, inline_label = match.group(1), match.group(2)
        name = inline_label or user_id_map.get(user_id) or f"user_{user_id}"
        return f"@{name}"

    text = _SPECIAL_MENTION_RE.sub(lambda m: f"@{m.group(1)}", text)
    return _MENTION_RE.sub(_sub, text)


def _resolve_channel_refs(text: str, channel_id_map: Optional[Dict[str, str]]) -> str:
    channel_id_map = channel_id_map or {}

    def _sub(match: re.Match) -> str:
        channel_id, inline_label = match.group(1), match.group(2)
        name = inline_label or channel_id_map.get(channel_id) or f"channel_{channel_id}"
        return f"#{name}"

    return _CHANNEL_REF_RE.sub(_sub, text)


def _resolve_links(text: str) -> str:
    def _sub(match: re.Match) -> str:
        url, label = match.group(1), match.group(2)
        return f"{label} ({url})" if label else url
    return _LINK_RE.sub(_sub, text)


def _strip_markdown_formatting(text: str) -> str:
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _STRIKE_RE.sub(r"\1", text)
    return text


def _convert_emoji_shortcodes(text: str) -> str:
    def _sub(match: re.Match) -> str:
        name = match.group(1).replace("_", " ").replace("-", " ")
        return name
    return _EMOJI_SHORTCODE_RE.sub(_sub, text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _normalize_elongation(text: str, max_repeats: int) -> str:
    pattern = re.compile(r"([a-zA-Z])\1{" + str(max_repeats) + r",}")
    return pattern.sub(lambda m: m.group(1) * max_repeats, text)


# ---------------------------------------------------------------------------
# SlackCleaner — implements the Cleaner contract
# ---------------------------------------------------------------------------

class SlackCleaner(Cleaner):
    """
    Cleans the text of a ``SlackMessage``.

    Handles all Slack-specific noise: mention/channel/link resolution,
    emoji shortcodes, markdown formatting, technical span protection,
    boilerplate detection, elongation normalization, and whitespace cleanup.
    """

    def __init__(
        self,
        config: Optional[SlackCleaningConfig] = None,
        user_id_map: Optional[Dict[str, str]] = None,
        channel_id_map: Optional[Dict[str, str]] = None,
    ):
        self.config = config or SlackCleaningConfig()
        self.user_id_map = user_id_map or {}
        self.channel_id_map = channel_id_map or {}

    def clean(self, source: SlackMessage) -> SlackMessage:
        """Return a new ``SlackMessage`` with cleaned text; all other fields preserved."""
        cfg = self.config
        raw_text = source.text or ""
        metadata = source.metadata or {}

        if cfg.drop_boilerplate and _is_boilerplate(raw_text, metadata):
            return source.model_copy(update={"text": ""})

        text = raw_text

        # Step 1: protect technical content before any other step runs.
        spans: List[str] = []
        if cfg.protect_technical_spans:
            text, spans = _protect_technical_spans(text)

        # Step 2: Slack structural syntax -> human-readable text.
        if cfg.resolve_mentions:
            text = _resolve_mentions(text, self.user_id_map)
        if cfg.resolve_channel_refs:
            text = _resolve_channel_refs(text, self.channel_id_map)
        if cfg.resolve_links:
            text = _resolve_links(text)
        if cfg.strip_markdown_formatting:
            text = _strip_markdown_formatting(text)
        if cfg.convert_emoji_shortcodes:
            text = _convert_emoji_shortcodes(text)
        if cfg.convert_unicode_emoji:
            text = convert_unicode_emoji(text)
        if cfg.expand_abbreviations:
            text = expand_abbreviations(text)

        # Step 3: noise reduction.
        if cfg.normalize_elongation:
            text = _normalize_elongation(text, cfg.max_repeated_chars)
        if cfg.normalize_whitespace:
            text = _normalize_whitespace(text)

        # Step 4: restore technical spans verbatim.
        if cfg.protect_technical_spans:
            text = _restore_technical_spans(text, spans)

        return source.model_copy(update={"text": text})


__all__ = ["SlackCleaner", "SlackCleaningConfig", "deduplicate_messages"]

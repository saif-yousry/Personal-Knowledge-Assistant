"""
cleaner_utils.py

Generic, reusable text-processing helpers for the preprocessing package.

Deliberately free of any email-specific or RAG-specific business logic
(no signature/reply-history heuristics live here) so these functions can
be reused by cleaners for other document types later (Slack messages,
PDFs, web pages, ...).
"""

from __future__ import annotations

import difflib
import hashlib
import html
import re
from html.parser import HTMLParser
from typing import Dict, List, Optional, TypeVar

import emoji

from models import BaseMessage

T = TypeVar("T", bound=BaseMessage)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

class _HTMLToTextParser(HTMLParser):
    """
    Minimal, dependency-free HTML-to-text converter.

    - Strips all tags, decodes HTML entities.
    - Drops the content of <script> and <style> elements entirely.
    - Drops the content of <blockquote> elements, since in HTML email
      these almost always wrap quoted reply history rather than the
      author's own message.
    - Inserts newlines at block-level tag boundaries so paragraphs and
      list items don't get glued together into one run-on line.
    """

    _BLOCK_TAGS = {
        "p", "div", "br", "li", "tr", "table",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
    _SKIP_CONTENT_TAGS = {"script", "style", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ARG002
        if tag in self._SKIP_CONTENT_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def get_text(self) -> str:
        """Return the accumulated plain text."""
        return "".join(self._chunks)


def html_to_text(raw_html: str) -> str:
    """Convert an HTML string to plain text (tags/scripts/styles/blockquotes stripped)."""
    if not raw_html:
        return ""
    parser = _HTMLToTextParser()
    parser.feed(raw_html)
    parser.close()
    return html.unescape(parser.get_text())


def looks_like_html(text: str) -> bool:
    """Heuristic check for whether a string contains HTML markup."""
    if not text:
        return False
    return bool(re.search(r"<\s*[a-zA-Z][^>]*>", text))


# ---------------------------------------------------------------------------
# Whitespace / normalization
# ---------------------------------------------------------------------------

def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs into one space, limit blank lines, and strip."""
    if not text:
        return ""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def strip_empty_lines(text: str, max_consecutive: int = 1) -> str:
    """Collapse runs of blank lines down to at most `max_consecutive`, and trim the edges."""
    if not text:
        return ""

    lines = text.split("\n")
    result: List[str] = []
    blank_run = 0

    for line in lines:
        if line.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                result.append("")
        else:
            blank_run = 0
            result.append(line)

    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------

ABBREVIATIONS: Dict[str, str] = {
    "afaik": "as far as I know",
    "brb": "be right back",
    "btw": "by the way",
    "fyi": "for your information",
    "idk": "I don't know",
    "ikr": "I know right",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "jk": "just kidding",
    "lol": "laughing out loud",
    "ngl": "not going to lie",
    "np": "no problem",
    "nvm": "never mind",
    "omg": "oh my god",
    "pls": "please",
    "plz": "please",
    "rn": "right now",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "thx": "thanks",
    "ttyl": "talk to you later",
    "ty": "thank you",
    "tysm": "thank you so much",
    "yw": "you're welcome",
}

ABBREVIATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./:@-])(" + "|".join(ABBREVIATIONS) + r")(?![A-Za-z0-9_./:@-])",
    re.IGNORECASE,
)


def _expand_abbreviation(match: re.Match) -> str:
    return ABBREVIATIONS[match.group(1).lower()]


def expand_abbreviations(text: str) -> str:
    """Expand common chat abbreviations to full phrases."""
    return ABBREVIATION_PATTERN.sub(_expand_abbreviation, text)


# ---------------------------------------------------------------------------
# Emoji helpers
# ---------------------------------------------------------------------------

def convert_unicode_emoji(text: str) -> str:
    """Convert Unicode emoji characters to readable text."""
    return emoji.demojize(text, delimiters=(" ", " ")).replace("_", " ")


# ---------------------------------------------------------------------------
# Hashing / deduplication
# ---------------------------------------------------------------------------

def hash_text(text: str) -> str:
    """Return a SHA-256 hash of whitespace-normalized, lowercased text."""
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_messages(
    messages: List[T],
    threshold: float = 0.92,
) -> List[T]:
    """
    Flag exact and near-duplicate messages in a batch.

    Returns a new list with ``is_duplicate_of`` set on duplicates.
    Original ordering is preserved.
    """
    seen_hashes: Dict[str, T] = {}
    results: List[T] = []

    for msg in messages:
        if not msg.text:
            results.append(msg)
            continue

        msg_hash = hash_text(msg.text)

        if msg_hash in seen_hashes:
            canonical = seen_hashes[msg_hash]
            results.append(msg.model_copy(update={"is_duplicate_of": canonical.id}))
            continue

        duplicate_of: Optional[str] = None
        for other in results:
            if other.is_duplicate_of is not None or not other.text:
                continue
            ratio = difflib.SequenceMatcher(None, msg.text, other.text).ratio()
            if ratio >= threshold:
                duplicate_of = other.id
                break

        seen_hashes[msg_hash] = msg
        if duplicate_of:
            results.append(msg.model_copy(update={"is_duplicate_of": duplicate_of}))
        else:
            results.append(msg)

    return results

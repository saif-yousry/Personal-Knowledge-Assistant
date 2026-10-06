"""
Chunk using LangChain's `RecursiveCharacterTextSplitter`.

Data flow:
    Email | Document  --(LangChainChunker)-->  List[Chunk]

Only splits text and attaches metadata.
"""

from __future__ import annotations

import os
import re
from typing import Callable, List, Optional, Union


from langchain_text_splitters import RecursiveCharacterTextSplitter


from models import DiscordMessage, Document, Email, SlackMessage, TelegramMessage


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class LangChainChunker():
    """
    Splits text into overlapping chunks using LangChain's
    `RecursiveCharacterTextSplitter`, preserving source metadata on every
    chunk. Accepts both Email and Document models.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS
        self.length_function = length_function

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=self.length_function,
        )

    def chunk(self, source: Union[Email, Document, SlackMessage, DiscordMessage, TelegramMessage]) -> List[dict]:
        """Split source text into chunks, preserving metadata."""

        doc_id, text, metadata = self._extract(source)
        text_pieces = self._splitter.split_text(text)

        chunks: List[dict] = []
        total_chunks = len(text_pieces)

        for index, piece in enumerate(text_pieces):
            if not piece.strip():
                continue

            chunk_obj = {
                "id": f"{doc_id}_chunk_{index}",
                "text": piece,
                "chunk_index": index,
                "metadata": {
                    **metadata,
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                },
            }
            chunks.append(chunk_obj)

        return chunks

    def _extract(self, source: Union[Email, Document, SlackMessage, DiscordMessage, TelegramMessage]) -> tuple[str, str, dict]:
        """Extract id, text, and metadata from the source."""
        if isinstance(source, Email):
            return (
                source.id,
                source.body or "",
                {
                    "source_type": "email",
                    "email_id": source.id,
                    "sender": source.sender,
                    "recipients": list(source.recipients),
                    "subject": source.subject,
                    "date": source.date.isoformat() if source.date else None,
                },
            )

        if isinstance(source, Document):
            base_name = os.path.splitext(source.source)[0]
            title = re.sub(r"^NovaTech_\d+_", "", base_name).replace("_", " ")
            return (
                f"{base_name}_p{source.page}",
                source.text,
                {
                    "source_type": "pdf",
                    "filename": source.source,
                    "title": title,
                    "page": source.page,
                    "document_type": source.doc_type,
                },
            )

        if isinstance(source, SlackMessage):
            return (
                source.id,
                source.text,
                {
                    "source_type": "slack",
                    "message_id": source.id,
                    "channel_id": source.channel_id,
                    "channel_name": source.channel_name,
                    "user_id": source.user_id,
                    "username": source.username,
                    "timestamp": source.timestamp.isoformat(),
                    "thread_ts": source.thread_ts,
                },
            )

        if isinstance(source, DiscordMessage):
            return (
                source.id,
                source.text,
                {
                    "source_type": "discord",
                    "message_id": source.id,
                    "guild_id": source.guild_id,
                    "channel_id": source.channel_id,
                    "channel_name": source.channel_name,
                    "author_id": source.author_id,
                    "author_name": source.author_name,
                    "timestamp": source.timestamp.isoformat(),
                    "reply_to": source.reply_to_message_id,
                },
            )

        if isinstance(source, TelegramMessage):
            return (
                source.id,
                source.text,
                {
                    "source_type": "telegram",
                    "message_id": source.id,
                    "chat_id": source.chat_id,
                    "chat_title": source.chat_title,
                    "chat_type": source.chat_type,
                    "user_id": source.user_id,
                    "username": source.username,
                    "timestamp": source.timestamp.isoformat(),
                    "reply_to": source.reply_to_message_id,
                },
            )

        raise TypeError(f"Unsupported source type: {type(source)}")

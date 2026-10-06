"""
pipeline.py

Orchestrates the RAG ingestion pipeline: clean -> chunk -> embed -> store.
Accepts both Email and Document objects through a unified interface.
"""

from __future__ import annotations

import logging
from typing import List, Union

from models import DiscordMessage, Document, Email, SlackMessage, TelegramMessage
from rag.preprocessing.cleaner_dispatcher import CleanerDispatcher
from rag.chunker import LangChainChunker
from rag.embedder import SentenceTransformerEmbedder
from rag.vector_store.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the RAG ingestion flow: clean, chunk, embed, store."""

    def __init__(
        self,
        store: ChromaVectorStore,
        embedder: SentenceTransformerEmbedder,
        dispatcher: CleanerDispatcher,
        chunker: LangChainChunker,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.dispatcher = dispatcher
        self.chunker = chunker

    def run(self, documents: List[Union[Email, Document, SlackMessage, DiscordMessage, TelegramMessage]]) -> int:
        """
        Run the full ingestion pipeline on a list of documents.
        Emails and PDFs can be mixed in the same list.

        Returns the total number of chunks stored.
        """
        cleaned = []
        for doc in documents:
            try:
                cleaned.append(self.dispatcher.clean(doc))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Skipping document due to cleaning error: %s", exc)
        logger.info("Cleaned %d/%d documents.", len(cleaned), len(documents))

        chunks = []
        for doc in cleaned:
            try:
                chunks.extend(self.chunker.chunk(doc))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("Skipping document due to chunking error: %s", exc)
        logger.info("Produced %d chunks.", len(chunks))

        if not chunks:
            logger.warning("No chunks produced — nothing to store.")
            return 0

        embedded = self.embedder.embed_many(chunks)
        logger.info("Generated %d embeddings.", len(embedded))

        self.store.add(embedded)
        logger.info("Stored %d embedded chunks.", len(embedded))

        return len(embedded)

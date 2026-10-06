"""Abstract base class defining the vector store contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorStore(ABC):
    """Abstract base class for all vector store implementations.

    Responsible only for storing already-embedded document chunks,
    retrieving similar chunks via embedding-based search, and persisting
    the index to disk.  Fully independent of loading, cleaning,
    chunking, prompts, and LLM calls.

    The pipeline feeds the store with pre-computed embedded chunks
    (output of ``SentenceTransformerEmbedder.embed_many()``):

    .. code-block:: python

        {
            "chunk":  {"id": str, "email_id": str, "text": str,
                       "chunk_index": int, "metadata": dict},
            "vector": [float, ...],
        }
    """

    @abstractmethod
    def add(self, embedded_chunks: List[Dict[str, Any]]) -> None:
        """Add a list of already-embedded chunk dicts to the store.

        Each dict must contain ``"chunk"`` (the chunk metadata dict from
        the chunker) and ``"vector"`` (the pre-computed embedding).
        Embedding is **not** recomputed inside this method.
        """

    @abstractmethod
    def similarity_search(self, query: str, k: int = 5, source_type: str | None = None) -> List[Dict[str, Any]]:
        """Return the top-*k* chunks most similar to *query*.

        The query is embedded using the store's ``embedding_function``;
        results are returned as plain dicts matching the original chunk
        schema (``id``, ``email_id``, ``text``, ``chunk_index``,
        ``metadata``).
        """

    @abstractmethod
    def delete(self, ids: Optional[List[str]] = None) -> None:
        """Delete documents by their IDs.

        If *ids* is ``None``, clear the entire collection.
        """

    @abstractmethod
    def persist(self) -> None:
        """Flush all pending changes to disk."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of indexed documents."""

"""
Embed using `sentence-transformers` library.

Data flow:
    List[Chunk]  --(SentenceTransformerEmbedder)-->  List[EmbeddedChunk]
"""

from __future__ import annotations

from typing import List

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentenceTransformerEmbedder:
    """
    Generates embeddings for `Chunk` objects using a local
    `sentence-transformers` model.

    The model is loaded exactly once, at construction time, and reused
    for every subsequent `embed_many()` call.

    Parameters
    ----------
    model_name:
        Name (or local path) of the sentence-transformers model to load.
        Defaults to "all-mpnet-base-v2".
    device:
        Device to run the model on, e.g. "cpu" or "cuda" or "auto". If omitted,
        sentence-transformers auto-selects the best available device.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        batch_size: int = 32
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self._model = SentenceTransformer(model_name, device=device)
        self.embedding_dimension: int = self._model.get_embedding_dimension()

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Run the sentence-transformers model over a batch of texts."""
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=self.batch_size
        )
        return [vector.tolist() for vector in embeddings]

    def embed_many(self, chunks: List[dict]) -> List[dict]:
        """
        Generate embeddings for many chunks in a single batched model
        call for better throughput than embedding one at a time.
        """
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        vectors = self.encode(texts)

        return [
            self._create_embedded_chunk(chunk, vector)
            for chunk, vector in zip(chunks, vectors)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_embedded_chunk(self, chunk: dict, vector: List[float]) -> dict:
        """Pair a chunk with its vector; the original chunk is stored unchanged."""
        return {
            "chunk": chunk,
            "vector": vector
        }

"""
Shared application dependencies.
Initialized once at import time and reused across routes and services.
"""

from config import settings

from database.postgresql import Database  # pylint: disable=wrong-import-position
from rag.embedder import SentenceTransformerEmbedder  # pylint: disable=wrong-import-position
from rag.chunker import LangChainChunker  # pylint: disable=wrong-import-position
from rag.preprocessing.cleaner_dispatcher import CleanerDispatcher  # pylint: disable=wrong-import-position
from rag.vector_store.chroma import ChromaVectorStore  # pylint: disable=wrong-import-position

db = Database(settings)
embedder = SentenceTransformerEmbedder()
store = ChromaVectorStore(embedder, settings.VECTOR_STORE_DIR, "knowledge")
chunker = LangChainChunker()
dispatcher = CleanerDispatcher()



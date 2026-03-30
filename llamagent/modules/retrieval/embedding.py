"""Embedding providers for the shared retrieval layer."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding computation."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string into a vector."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class ChromaDefaultEmbedding(EmbeddingProvider):
    """Wraps ChromaDB's built-in embedding function (all-MiniLM-L6-v2). Zero extra dependencies."""

    def __init__(self):
        # Lazy init — only import chromadb when first used
        self._fn = None
        self._dims = 384  # all-MiniLM-L6-v2 output dimension

    def _ensure_fn(self):
        if self._fn is not None:
            return
        try:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._fn = DefaultEmbeddingFunction()
        except ImportError:
            raise ImportError(
                "chromadb is required for ChromaDefaultEmbedding. "
                "Install: pip install chromadb"
            )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using ChromaDB's default embedding function."""
        self._ensure_fn()
        return self._fn(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    @property
    def dimensions(self) -> int:
        """Return 384 (all-MiniLM-L6-v2 output dimension)."""
        return self._dims

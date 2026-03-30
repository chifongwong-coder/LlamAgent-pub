"""Vector storage backends for the shared retrieval layer."""

from abc import ABC, abstractmethod


class VectorBackend(ABC):
    """Abstract vector storage. Receives pre-computed embeddings from EmbeddingProvider."""

    @abstractmethod
    def save(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        """Save a document with its pre-computed embedding."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int, filters: dict | None = None) -> list[dict]:
        """Search for similar documents by embedding vector."""
        ...

    @abstractmethod
    def get(self, id: str) -> dict | None:
        """Get a document by ID. Returns {"id", "text", "metadata"} or None if not found."""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        ...

    @abstractmethod
    def update(self, id: str, text: str | None = None, embedding: list[float] | None = None,
               metadata: dict | None = None) -> None:
        """Update a document's text, embedding, and/or metadata."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        ...


class ChromaVectorBackend(VectorBackend):
    """ChromaDB-based vector backend. Uses embeddings= parameter (no internal embedding)."""

    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._available = True

    def _ensure_collection(self):
        """Lazily initialize the ChromaDB client and collection."""
        if self._collection is not None:
            return
        if not self._available:
            raise RuntimeError("ChromaDB is not available")
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError:
            self._available = False
            raise ImportError("chromadb required. Install: pip install chromadb")

    def save(self, id: str, text: str, embedding: list[float], metadata: dict) -> None:
        """Save (upsert) a document with its pre-computed embedding."""
        self._ensure_collection()
        self._collection.upsert(
            ids=[id], documents=[text], embeddings=[embedding], metadatas=[metadata]
        )

    def search(self, query_embedding: list[float], top_k: int,
               filters: dict | None = None) -> list[dict]:
        """Search by cosine similarity. Returns list of {id, text, metadata, score}."""
        self._ensure_collection()
        count = self._collection.count()
        if count == 0:
            return []
        where = filters if filters else None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
        )
        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                items.append({
                    "id": results["ids"][0][i],
                    "text": doc,
                    "metadata": meta,
                    "score": round(1 / (1 + dist), 4),
                })
        return items

    def get(self, id: str) -> dict | None:
        """Get a document by ID. Returns None if not found."""
        self._ensure_collection()
        try:
            result = self._collection.get(ids=[id])
            if result and result["ids"] and result["ids"][0]:
                meta = result["metadatas"][0] if result["metadatas"] else {}
                text = result["documents"][0] if result["documents"] else ""
                return {"id": result["ids"][0], "text": text, "metadata": meta}
        except Exception:
            pass
        return None

    def delete(self, id: str) -> None:
        """Delete a document by ID. Silently ignores missing IDs."""
        self._ensure_collection()
        try:
            self._collection.delete(ids=[id])
        except Exception:
            pass  # Ignore if ID doesn't exist

    def update(self, id: str, text: str | None = None, embedding: list[float] | None = None,
               metadata: dict | None = None) -> None:
        """Update specific fields of a document."""
        self._ensure_collection()
        kwargs = {"ids": [id]}
        if text is not None:
            kwargs["documents"] = [text]
        if embedding is not None:
            kwargs["embeddings"] = [embedding]
        if metadata is not None:
            kwargs["metadatas"] = [metadata]
        self._collection.update(**kwargs)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        self._ensure_collection()
        return self._collection.count()

    def clear(self) -> None:
        """Drop and recreate the collection to remove all documents."""
        self._ensure_collection()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

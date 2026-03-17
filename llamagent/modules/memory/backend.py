"""
Memory storage backend abstract interface.

Whether the underlying storage uses JSON, SQLite, or ChromaDB doesn't matter,
as long as it implements the four methods: save / search / clear / stats.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class MemoryBackend(ABC):
    """Memory backend abstract base class. All persistence implementations inherit from this."""

    @abstractmethod
    def save(self, content: str, category: str = "conversation", metadata: dict | None = None):
        """Save a memory entry."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve relevant memories.

        Returns:
            [{"content": str, "category": str, "created_at": str, "score": float}, ...]
        """
        ...

    @abstractmethod
    def clear(self):
        """Clear all memories."""
        ...

    @abstractmethod
    def stats(self) -> dict:
        """Return backend statistics."""
        ...


class ChromaMemoryBackend(MemoryBackend):
    """
    ChromaDB-based memory backend with semantic retrieval support.

    chromadb is an optional dependency with graceful degradation when not installed:
    - Lazy initialization (imports chromadb only on first call)
    - On ImportError, marks _available=False; subsequent calls silently degrade
    - Relevance formula: round(1 / (1 + distance), 4), range (0, 1]
    """

    def __init__(self, persist_dir: str, collection_name: str = "llamagent_memory"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._available = True

    def _ensure_client(self) -> bool:
        """Lazily initialize ChromaDB. Imports on first call; marks unavailable on failure."""
        if self._client is not None:
            return True
        if not self._available:
            return False

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            return True
        except ImportError:
            print("[Memory] chromadb is not installed, long-term memory is unavailable. Please run: pip install chromadb")
            self._available = False
            return False
        except Exception as e:
            print(f"[Memory] ChromaDB initialization failed: {e}")
            self._available = False
            return False

    def save(self, content: str, category: str = "conversation", metadata: dict | None = None):
        """Save a memory entry to ChromaDB. Silently skips when chromadb is unavailable."""
        if not self._ensure_client():
            return

        import hashlib
        now = datetime.now().isoformat()
        # Generate unique ID from content + timestamp to avoid overwriting memories with same content at different times
        memory_id = hashlib.md5(f"{content}_{now}".encode()).hexdigest()

        meta = {"category": category, "created_at": now}
        if metadata:
            meta.update(metadata)

        try:
            self._collection.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[meta],
            )
        except Exception as e:
            print(f"[Memory] Save failed: {e}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic retrieval of relevant memories.

        Returns:
            [{"content": str, "category": str, "created_at": str, "score": float}, ...]
            Returns an empty list when chromadb is unavailable or no results are found.
        """
        if not self._ensure_client() or self._collection is None:
            return []

        try:
            count = self._collection.count()
            if count == 0:
                return []

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
        except Exception as e:
            print(f"[Memory] Search failed: {e}")
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                items.append({
                    "content": doc,
                    "category": meta.get("category", ""),
                    "created_at": meta.get("created_at", ""),
                    "score": round(1 / (1 + dist), 4),
                })

        return items

    def clear(self):
        """Clear all memories. Silently skips when chromadb is unavailable."""
        if not self._ensure_client():
            return
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
        except Exception as e:
            print(f"[Memory] Clear failed: {e}")

    def stats(self) -> dict:
        """Return backend statistics."""
        if not self._ensure_client():
            return {"available": False, "count": 0}

        try:
            return {
                "available": True,
                "count": self._collection.count(),
                "persist_dir": self.persist_dir,
                "collection": self.collection_name,
            }
        except Exception as e:
            return {"available": False, "count": 0, "error": str(e)}

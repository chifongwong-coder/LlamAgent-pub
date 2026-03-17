"""
Memory Store Manager: unified management interface for long-term memory.

Long-term memory persistence is handled by MemoryBackend (currently ChromaMemoryBackend).
Short-term memory (conversation context) is managed by the core's conversation history and is not part of this module.
"""

from llamagent.modules.memory.backend import MemoryBackend


class MemoryStore:
    """
    Memory Store Manager.

    Responsibilities:
    - Long-term memory: delegated to MemoryBackend implementation (ChromaDB / JSON / SQLite ...)
    - Provides four core methods: save_memory / recall / clear_long_term / get_stats
    """

    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    # ============================================================
    # Long-term memory (delegated to backend)
    # ============================================================

    def save_memory(self, content: str, category: str = "conversation"):
        """Save a memory entry to the backend."""
        self.backend.save(content, category)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic retrieval of relevant memories from the backend."""
        return self.backend.search(query, top_k)

    def clear_long_term(self):
        """Clear all long-term memories."""
        self.backend.clear()

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return self.backend.stats()

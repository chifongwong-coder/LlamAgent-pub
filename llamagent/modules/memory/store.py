"""
Memory Store Manager: unified management interface for structured long-term memory.

v1.7 rewrite: manages MemoryFacts via RetrievalPipeline.
Provides fact-level CRUD, key-based lookup for the merger, and legacy-compatible
save_memory/recall methods.
"""

import hashlib
import logging
from datetime import datetime

from llamagent.modules.memory.fact import MemoryFact

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Memory Store Manager built on top of RetrievalPipeline.

    Responsibilities:
    - save_fact(fact): persist a MemoryFact via the retrieval pipeline
    - search_facts(query, top_k, status_filter): semantic search with optional status filter
    - get_facts_by_key(kind, subject, attribute): exact key lookup for merger
    - update_fact_status(fact_id, status): change a fact's lifecycle status
    - update_fact_accessed(fact_id): update last_accessed_at timestamp and reinforce strength
    - clear(): remove all stored facts
    - get_stats(): return memory statistics
    - Legacy methods: save_memory / recall / clear_long_term (backward compatibility)
    """

    def __init__(self, pipeline=None):
        """
        Initialize MemoryStore.

        Args:
            pipeline: A RetrievalPipeline instance (from factory). None = unavailable.
        """
        self._pipeline = pipeline

    @property
    def available(self) -> bool:
        """Check if the store has a usable storage backend."""
        return self._pipeline is not None

    # ============================================================
    # Fact-level operations (v1.7)
    # ============================================================

    def save_fact(self, fact: MemoryFact) -> None:
        """
        Persist a MemoryFact to the retrieval pipeline.

        The fact's to_text() is used as the embeddable document, and
        to_metadata() provides structured metadata for filtering.

        Args:
            fact: The MemoryFact to save.
        """
        if self._pipeline is not None:
            self._pipeline.save(
                id=fact.fact_id,
                text=fact.to_text(),
                metadata=fact.to_metadata(),
            )
        else:
            logger.warning("[Memory] No storage backend available; fact not saved")

    def search_facts(
        self, query: str, top_k: int = 5, status_filter: str = "active"
    ) -> list[dict]:
        """
        Semantic search for facts matching a query.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            status_filter: Only return facts with this status ("active" by default).
                          Set to None or "" to return all statuses.

        Returns:
            List of dicts with keys: id, text, metadata, score.
        """
        if self._pipeline is None:
            return []

        try:
            results = self._pipeline.search(
                query=query,
                top_k=top_k * 2,  # Fetch extra to account for post-filtering
                mode="vector",
            )
        except Exception as e:
            logger.warning("[Memory] Pipeline search failed: %s", e)
            return []

        # Apply status filter client-side
        if status_filter:
            results = [
                r for r in results
                if r.get("metadata", {}).get("status", "active") == status_filter
            ]

        return results[:top_k]

    def get_facts_by_key(
        self, kind: str, subject: str, attribute: str
    ) -> list[dict]:
        """
        Look up existing facts by exact (kind, subject, attribute) triple.

        Used by FactMerger for deduplication. Searches by the fact's text
        representation and then filters results by exact metadata match.

        Args:
            kind: Fact kind (e.g. "preference").
            subject: Normalized subject key.
            attribute: Normalized attribute key.

        Returns:
            List of matching fact metadata dicts.
        """
        if self._pipeline is None:
            return []

        # Search using the key text to get candidates, then filter exactly
        search_text = f"{kind}: {subject}.{attribute}"
        try:
            results = self._pipeline.search(
                query=search_text,
                top_k=10,
                mode="vector",
            )
        except Exception as e:
            logger.warning("[Memory] Key lookup failed: %s", e)
            return []

        matches = []
        for r in results:
            meta = r.get("metadata", {})
            if (
                meta.get("kind") == kind
                and meta.get("subject") == subject
                and meta.get("attribute") == attribute
            ):
                matches.append(meta)
        return matches

    def update_fact_status(self, fact_id: str, status: str) -> None:
        """
        Update the lifecycle status of an existing fact.

        Uses read-then-write to preserve all metadata fields (ChromaDB
        replaces entire metadata on update).

        Args:
            fact_id: The fact_id to update.
            status: New status ("active", "superseded", or "archived").
        """
        if self._pipeline is not None:
            try:
                doc = self._pipeline.vector.get(fact_id)
                if not doc:
                    return
                meta = doc.get("metadata", {})
                meta["status"] = status
                meta["updated_at"] = datetime.now().isoformat()
                self._pipeline.vector.update(id=fact_id, metadata=meta)
            except Exception as e:
                logger.warning("[Memory] Status update failed for %s: %s", fact_id, e)

    def update_fact_accessed(self, fact_id: str, increment_strength: float = 0.1, max_strength: float = 2.0) -> None:
        """
        Update the last_accessed_at timestamp and reinforce strength for a fact.

        Args:
            fact_id: The fact_id to update.
            increment_strength: Amount to increase strength by (default 0.1).
            max_strength: Maximum strength value (default 2.0).
        """
        if self._pipeline is not None:
            try:
                # Read current strength via direct ID lookup (no embedding needed)
                doc = self._pipeline.vector.get(fact_id)
                current_strength = 1.0
                if doc:
                    current_strength = doc.get("metadata", {}).get("strength", 1.0)
                    if isinstance(current_strength, str):
                        current_strength = float(current_strength)

                new_strength = min(current_strength + increment_strength, max_strength)
                self._pipeline.vector.update(
                    id=fact_id,
                    metadata={
                        "last_accessed_at": datetime.now().isoformat(),
                        "strength": new_strength,
                    },
                )
            except Exception as e:
                logger.warning("[Memory] Access update failed for %s: %s", fact_id, e)

    def list_all_active_facts(self) -> list[dict]:
        """Return metadata dicts for all active facts.

        Returns:
            List of metadata dicts (each dict has fact_id, kind, subject, etc.).
        """
        if self._pipeline is None:
            return []
        try:
            results = self._pipeline.vector.get_all(where={"status": "active"})
            return [r["metadata"] for r in results]
        except Exception as e:
            logger.warning("[Memory] list_all_active_facts failed: %s", e)
            return []

    def update_fact_value(self, fact_id: str, new_value: str) -> None:
        """Update a fact's value and updated_at timestamp.

        Uses read-then-write to preserve all metadata fields (ChromaDB
        replaces entire metadata on update).

        Args:
            fact_id: The fact_id to update.
            new_value: New value string for the fact.
        """
        if self._pipeline is not None:
            try:
                doc = self._pipeline.vector.get(fact_id)
                if not doc:
                    return
                meta = doc.get("metadata", {})
                meta["value"] = new_value
                meta["updated_at"] = datetime.now().isoformat()
                self._pipeline.vector.update(id=fact_id, metadata=meta)
            except Exception as e:
                logger.warning("[Memory] Value update failed for %s: %s", fact_id, e)

    def clear(self) -> None:
        """Clear all stored facts."""
        if self._pipeline is not None:
            try:
                self._pipeline.clear()
            except Exception as e:
                logger.warning("[Memory] Clear failed: %s", e)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        if self._pipeline is not None:
            try:
                count = self._pipeline.vector.count()
                return {"available": True, "count": count}
            except Exception as e:
                return {"available": False, "count": 0, "error": str(e)}
        return {"available": False, "count": 0}

    # ============================================================
    # Legacy interface (backward compatibility)
    # ============================================================

    def save_memory(self, content: str, category: str = "conversation") -> None:
        """Save a memory entry (legacy interface)."""
        if self._pipeline is not None:
            now = datetime.now().isoformat()
            memory_id = hashlib.md5(f"{content}_{now}".encode()).hexdigest()
            self._pipeline.save(
                id=memory_id,
                text=content,
                metadata={"category": category, "created_at": now, "status": "active"},
            )

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Semantic retrieval of relevant memories (legacy interface)."""
        if self._pipeline is None:
            return []

        try:
            results = self._pipeline.search(query=query, top_k=top_k, mode="vector")
        except Exception as e:
            logger.warning("[Memory] Recall failed: %s", e)
            return []

        # Convert pipeline results to legacy format
        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append({
                "content": r.get("text", r.get("content", "")),
                "category": meta.get("category", meta.get("kind", "")),
                "created_at": meta.get("created_at", ""),
                "score": r.get("score", 0.0),
            })
        return items

    def clear_long_term(self) -> None:
        """Clear all long-term memories (legacy alias for clear())."""
        self.clear()

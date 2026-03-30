"""Hybrid retrieval pipeline combining vector search, lexical search, and reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamagent.modules.retrieval.embedding import EmbeddingProvider
    from llamagent.modules.retrieval.lexical import LexicalBackend
    from llamagent.modules.retrieval.reranker import Reranker
    from llamagent.modules.retrieval.vector import VectorBackend


class RetrievalPipeline:
    """Combines vector + lexical search with RRF merging and optional reranking."""

    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector: VectorBackend,
        lexical: LexicalBackend | None = None,
        reranker: Reranker | None = None,
    ):
        """
        Args:
            embedding: EmbeddingProvider instance for computing query embeddings.
            vector: VectorBackend instance for similarity search.
            lexical: LexicalBackend instance (optional, needed for hybrid mode).
            reranker: Reranker instance (optional, applied after merging).
        """
        self.embedding = embedding
        self.vector = vector
        self.lexical = lexical
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
        """
        Execute a retrieval search.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            mode: One of "vector", "lexical", or "hybrid".

        Returns:
            List of {"id", "text", "metadata", "score"} dicts, ordered by relevance.
        """
        if mode == "vector" or (mode == "hybrid" and self.lexical is None):
            return self._vector_search(query, top_k)
        elif mode == "lexical":
            if self.lexical is None:
                return []
            return self.lexical.search(query, top_k)
        else:  # hybrid
            return self._hybrid_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Pure vector similarity search."""
        query_embedding = self.embedding.embed_query(query)
        return self.vector.search(query_embedding, top_k)

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """Hybrid search: vector + lexical with RRF merge and optional reranking."""
        # Fetch larger candidate sets from both sources
        fetch_k = max(top_k * 4, 20)

        vector_results = self._vector_search(query, fetch_k)
        lexical_results = self.lexical.search(query, fetch_k) if self.lexical else []

        # RRF (Reciprocal Rank Fusion) merge
        merged = self._rrf_merge(vector_results, lexical_results)

        # Optional reranking
        if self.reranker and merged:
            try:
                merged = self.reranker.rerank(query, merged, top_k)
            except Exception:
                # Rerank failed — fall back to RRF ordering
                merged = merged[:top_k]
        else:
            merged = merged[:top_k]

        return merged

    @staticmethod
    def _rrf_merge(
        vector_results: list[dict],
        lexical_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion: combine two ranked lists into one.

        Uses the formula: score(d) = sum(1 / (k + rank + 1)) across all lists
        where k is a smoothing constant (default 60).
        """
        scores: dict[str, float] = {}  # id -> rrf_score
        docs: dict[str, dict] = {}     # id -> doc dict

        for rank, doc in enumerate(vector_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            docs[doc_id] = doc

        for rank, doc in enumerate(lexical_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
            if doc_id not in docs:
                docs[doc_id] = doc

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for doc_id in sorted_ids:
            doc = docs[doc_id].copy()
            doc["score"] = round(scores[doc_id], 6)
            result.append(doc)

        return result

    def save(self, id: str, text: str, metadata: dict) -> None:
        """Save a document to both vector and lexical backends."""
        embedding = self.embedding.embed_texts([text])[0]
        self.vector.save(id, text, embedding, metadata)
        if self.lexical:
            try:
                self.lexical.index(id, text, metadata)
            except Exception:
                pass  # Best-effort: lexical index failure doesn't block vector save

    def delete(self, id: str) -> None:
        """Delete a document from both backends."""
        self.vector.delete(id)
        if self.lexical:
            try:
                self.lexical.delete(id)
            except Exception:
                pass

    def clear(self) -> None:
        """Clear all documents from both backends."""
        self.vector.clear()
        if self.lexical:
            try:
                self.lexical.clear()
            except Exception:
                pass

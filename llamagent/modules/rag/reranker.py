"""Reranking strategies for the RAG backend layer."""

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for result reranking."""

    @abstractmethod
    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """
        Rerank a list of documents by relevance to the query.

        Args:
            query: The original search query.
            documents: List of {"id", "text", "metadata", "score"} dicts.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of document dicts, truncated to top_k.
        """
        ...


class LLMReranker(Reranker):
    """Lightweight reranker using the agent's LLM. Falls back to original order on failure."""

    RERANK_SYSTEM = (
        "You are a relevance judge. Given a query and a list of documents, "
        "rank them by relevance to the query. Return a JSON array of document indices "
        "(0-based) sorted from most relevant to least relevant. "
        "Only return the JSON array, nothing else. Example: [2, 0, 4, 1, 3]"
    )

    def __init__(self, llm):
        """
        Args:
            llm: LLMClient instance with an ask_json method.
        """
        self.llm = llm

    def rerank(self, query: str, documents: list[dict], top_k: int) -> list[dict]:
        """
        Rerank documents using the LLM as a relevance judge.

        Sends the query and truncated document texts to the LLM, which returns
        a ranked list of indices. Falls back to original order on any failure.
        """
        if not documents:
            return []

        # Build prompt with numbered documents
        doc_list = []
        for i, doc in enumerate(documents):
            text = doc.get("text", "")[:300]  # Truncate for prompt size
            doc_list.append(f"[{i}] {text}")

        prompt = f"Query: {query}\n\nDocuments:\n" + "\n".join(doc_list)

        try:
            result = self.llm.ask_json(
                prompt=prompt,
                system=self.RERANK_SYSTEM,
                temperature=0,
            )

            # Parse result — expect a list of indices
            if isinstance(result, list):
                indices = result
            elif isinstance(result, dict):
                # Handle {"ranking": [...]}, {"indices": [...]}, etc.
                indices = None
                for key in ("ranking", "indices", "order", "result"):
                    if key in result and isinstance(result[key], list):
                        indices = result[key]
                        break
                if indices is None:
                    raise ValueError("No index list found in response")
            else:
                raise ValueError(f"Unexpected response type: {type(result)}")

            # Validate and reorder
            valid_indices = [
                i for i in indices if isinstance(i, int) and 0 <= i < len(documents)
            ]

            if not valid_indices:
                raise ValueError("No valid indices in response")

            # Add any missing indices at the end (in original order)
            seen = set(valid_indices)
            for i in range(len(documents)):
                if i not in seen:
                    valid_indices.append(i)

            reranked = [documents[i] for i in valid_indices[:top_k]]
            return reranked

        except Exception as e:
            logger.debug("LLMReranker failed, falling back to original order: %s", e)
            return documents[:top_k]

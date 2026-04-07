"""
Factory for creating retrieval components from configuration.

Modules (Memory, Retrieval, Reflection) should use this factory instead of
directly importing concrete implementation classes. This keeps modules
decoupled from specific backend implementations.

Usage:
    from llamagent.modules.rag.factory import create_pipeline

    pipeline = create_pipeline(
        config=agent.config,
        collection_name="memory_default",
        enable_lexical=False,
        llm=self.llm,  # for LLMReranker, optional
    )
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamagent.core.llm import LLMClient
    from llamagent.modules.rag.embedding import EmbeddingProvider
    from llamagent.modules.rag.vector import VectorBackend
    from llamagent.modules.rag.lexical import LexicalBackend
    from llamagent.modules.rag.pipeline import RetrievalPipeline
    from llamagent.modules.rag.reranker import Reranker

logger = logging.getLogger(__name__)


def create_embedding(config) -> "EmbeddingProvider":
    """
    Create an EmbeddingProvider based on config.embedding_provider.

    Currently supported:
    - "chromadb": ChromaDefaultEmbedding (all-MiniLM-L6-v2, requires chromadb)

    Raises:
        ImportError: If the required backend is not installed.
        ValueError: If the provider name is not recognized.
    """
    provider = getattr(config, "embedding_provider", "chromadb")

    if provider == "chromadb":
        from llamagent.modules.rag.embedding import ChromaDefaultEmbedding
        return ChromaDefaultEmbedding()
    else:
        raise ValueError(
            f"Unknown embedding_provider: '{provider}'. "
            f"Supported: 'chromadb'. "
            f"Future: 'openai', 'bge', etc."
        )


def create_vector_backend(config, collection_name: str) -> "VectorBackend":
    """
    Create a VectorBackend based on config.

    Currently supported:
    - ChromaVectorBackend (default, requires chromadb)

    Args:
        config: Agent config object.
        collection_name: Collection/namespace for this backend instance.
    """
    persist_dir = getattr(config, "retrieval_persist_dir",
                          getattr(config, "chroma_dir", "data/chroma"))

    from llamagent.modules.rag.vector import ChromaVectorBackend
    return ChromaVectorBackend(persist_dir=persist_dir, collection_name=collection_name)


def create_lexical_backend(config, name: str) -> "LexicalBackend":
    """
    Create a LexicalBackend based on config.

    Currently supported:
    - SQLiteFTSBackend (default, zero external dependencies)

    Args:
        config: Agent config object.
        name: Base name for the database file (e.g., "rag_fts").
    """
    persist_dir = getattr(config, "retrieval_persist_dir",
                          getattr(config, "chroma_dir", "data/chroma"))
    os.makedirs(persist_dir, exist_ok=True)
    db_path = os.path.join(persist_dir, f"{name}.db")

    from llamagent.modules.rag.lexical import SQLiteFTSBackend
    return SQLiteFTSBackend(db_path=db_path)


def create_reranker(config, llm: "LLMClient | None" = None) -> "Reranker | None":
    """
    Create a Reranker if enabled in config.

    Returns None if reranking is disabled or LLM is not available.

    Args:
        config: Agent config object.
        llm: LLMClient instance (required for LLMReranker).
    """
    enabled = getattr(config, "rag_rerank_enabled", False)
    if not enabled or llm is None:
        return None

    from llamagent.modules.rag.reranker import LLMReranker
    return LLMReranker(llm)


def create_pipeline(
    config,
    collection_name: str,
    enable_lexical: bool = False,
    lexical_name: str | None = None,
    llm: "LLMClient | None" = None,
    enable_reranker: bool = False,
) -> "RetrievalPipeline":
    """
    Create a fully assembled RetrievalPipeline from config.

    This is the primary entry point for modules. Modules should call this
    instead of directly importing concrete backend classes.

    Args:
        config: Agent config object.
        collection_name: Vector collection name (e.g., "memory_default", "llamagent_docs").
        enable_lexical: Whether to create a lexical backend for hybrid search.
        lexical_name: Base name for the FTS database file (defaults to collection_name + "_fts").
        llm: LLMClient for LLMReranker (optional).
        enable_reranker: Whether to create a reranker.

    Returns:
        An assembled RetrievalPipeline instance.

    Raises:
        ImportError: If required backends are not installed.
    """
    from llamagent.modules.rag.pipeline import RetrievalPipeline

    embedding = create_embedding(config)
    vector = create_vector_backend(config, collection_name)

    lexical = None
    if enable_lexical:
        try:
            lname = lexical_name or f"{collection_name}_fts"
            lexical = create_lexical_backend(config, lname)
        except Exception as e:
            logger.warning("Failed to create lexical backend: %s (hybrid search disabled)", e)

    reranker = None
    if enable_reranker:
        try:
            reranker = create_reranker(config, llm)
        except Exception as e:
            logger.warning("Failed to create reranker: %s (reranking disabled)", e)

    return RetrievalPipeline(
        embedding=embedding,
        vector=vector,
        lexical=lexical,
        reranker=reranker,
    )

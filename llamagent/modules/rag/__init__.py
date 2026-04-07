"""
RAG backend: shared retrieval infrastructure for Memory, Retrieval, and Reflection modules.

This is a backend package, NOT a pluggable Module. It provides:
- EmbeddingProvider: abstract embedding computation
- VectorBackend: abstract vector storage and retrieval
- LexicalBackend: abstract keyword search (BM25 via SQLite FTS5)
- RetrievalPipeline: hybrid search orchestration (vector + lexical + rerank)
- Reranker: abstract result reranking
- Factory functions: create_pipeline, create_embedding, etc.
- DocumentChunker: format-aware document splitting
- RAGRetriever: document loading and search via pipeline

Modules should use the factory functions (not concrete classes) to create
retrieval components. This keeps modules decoupled from specific backends.

Usage:
    from llamagent.modules.rag.factory import create_pipeline

    pipeline = create_pipeline(config=agent.config, collection_name="my_collection")
"""

# Abstract interfaces (for type hints and isinstance checks)
from llamagent.modules.rag.embedding import EmbeddingProvider
from llamagent.modules.rag.vector import VectorBackend
from llamagent.modules.rag.lexical import LexicalBackend
from llamagent.modules.rag.pipeline import RetrievalPipeline
from llamagent.modules.rag.reranker import Reranker

# Factory functions (primary entry point for modules)
from llamagent.modules.rag.factory import (
    create_pipeline,
    create_embedding,
    create_vector_backend,
    create_lexical_backend,
    create_reranker,
)

# Document processing
from llamagent.modules.rag.chunker import Chunk, DocumentChunker
from llamagent.modules.rag.retriever import RAGRetriever

__all__ = [
    "EmbeddingProvider",
    "VectorBackend",
    "LexicalBackend",
    "RetrievalPipeline",
    "Reranker",
    "create_pipeline",
    "create_embedding",
    "create_vector_backend",
    "create_lexical_backend",
    "create_reranker",
    "Chunk",
    "DocumentChunker",
    "RAGRetriever",
]

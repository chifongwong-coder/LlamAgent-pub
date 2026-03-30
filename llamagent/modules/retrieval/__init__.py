"""
Shared retrieval infrastructure for Memory, RAG, and Reflection modules.

This is a service package, NOT a pluggable Module. It provides:
- EmbeddingProvider: abstract embedding computation
- VectorBackend: abstract vector storage and retrieval
- LexicalBackend: abstract keyword search (BM25 via SQLite FTS5)
- RetrievalPipeline: hybrid search orchestration (vector + lexical + rerank)
- Reranker: abstract result reranking
- Factory functions: create_pipeline, create_embedding, etc.

Modules should use the factory functions (not concrete classes) to create
retrieval components. This keeps modules decoupled from specific backends.

Usage:
    from llamagent.modules.retrieval.factory import create_pipeline

    pipeline = create_pipeline(config=agent.config, collection_name="my_collection")
"""

# Abstract interfaces (for type hints and isinstance checks)
from llamagent.modules.retrieval.embedding import EmbeddingProvider
from llamagent.modules.retrieval.vector import VectorBackend
from llamagent.modules.retrieval.lexical import LexicalBackend
from llamagent.modules.retrieval.pipeline import RetrievalPipeline
from llamagent.modules.retrieval.reranker import Reranker

# Factory functions (primary entry point for modules)
from llamagent.modules.retrieval.factory import (
    create_pipeline,
    create_embedding,
    create_vector_backend,
    create_lexical_backend,
    create_reranker,
)

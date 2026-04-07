"""
Retrieval Module: knowledge retrieval over documents from the knowledge base.

Supports two backends selected via config.retrieval_backend:
- RAG (default): vector/lexical/hybrid search via llamagent.modules.rag
- FS: file-system based document browsing via llamagent.modules.fs_store

Usage:
    from llamagent.modules.retrieval import RetrievalModule

    agent.register_module(RetrievalModule())
"""

from llamagent.modules.retrieval.module import RetrievalModule

__all__ = ["RetrievalModule"]

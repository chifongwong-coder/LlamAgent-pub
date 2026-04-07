"""
Retrieval Module: provides knowledge retrieval augmentation based on private documents for LlamAgent.

Supports two backends:
- RAG (default): vector/lexical search via ChromaDB. Registers search_knowledge tool.
- FS: file-system based browsing of markdown documents. Registers list_knowledge,
  list_entries, and read_entry tools for step-by-step document navigation.

Tool-based (Agentic) mode:
- on_attach registers tools based on backend; the model autonomously decides when to retrieve
- on_context injects an appropriate usage guide
- Graceful degradation when chromadb is not installed (RAG backend)
"""

import logging
import os

from llamagent.core.agent import Module

logger = logging.getLogger(__name__)

# Knowledge base usage guide injected into context, letting the model know it has retrieval capabilities
RAG_GUIDE = """\
[Knowledge Base] You can search the local knowledge base for professional materials.
- Use the search_knowledge tool to search relevant documents (when the user asks about specialized knowledge or needs reference materials)
- Search results may contain multiple entries; filter for the most relevant ones
- For common knowledge questions, answer directly without searching the knowledge base"""

FS_RETRIEVE_GUIDE = """\
[Knowledge Base] You can browse documents in the local knowledge base.
- Use list_knowledge to see all available documents and their descriptions.
- Use list_entries to see the sections within a specific document.
- Use read_entry to read the full content of a specific section.
- Browse step by step: documents -> entries -> content."""


class RetrievalModule(Module):
    """
    Retrieval Module: knowledge retrieval over documents.

    Supports RAG and FS backends. Backend is selected via config.retrieval_backend.
    - RAG: search_knowledge tool backed by ChromaDB vector search
    - FS: three-step tool chain (list_knowledge, list_entries, read_entry)
    """

    name: str = "retrieval"
    description: str = "Knowledge retrieval: search and browse documents from knowledge base"

    def __init__(self):
        self.retriever = None  # RAG backend only
        self._fs_store = None  # FS backend only
        self._backend: str = "rag"

    def on_attach(self, agent):
        """Initialize the retrieval backend and register tools."""
        super().on_attach(agent)

        self._backend = getattr(agent.config, "retrieval_backend", "rag")

        if self._backend == "fs":
            self._init_fs_backend(agent)
        else:
            self._init_rag_backend(agent)

    def _init_rag_backend(self, agent):
        """Initialize RAG backend: pipeline + chunker + retriever + search tool."""
        from llamagent.modules.rag.chunker import DocumentChunker
        from llamagent.modules.rag.retriever import RAGRetriever

        cfg = agent.config

        # Build the retrieval pipeline components
        pipeline = self._build_pipeline(agent)
        if pipeline is None:
            logger.warning("[Retrieval] Pipeline initialization failed; retrieval will be unavailable")

        # Create chunker and retriever
        chunker = DocumentChunker(chunk_size=cfg.chunk_size)
        self.retriever = RAGRetriever(
            pipeline=pipeline,
            chunker=chunker,
            top_k=cfg.rag_top_k,
            mode=cfg.rag_retrieval_mode,
        )

        # Register search_knowledge tool
        self._register_tools()

    def _init_fs_backend(self, agent):
        """Initialize FS backend: FSStore + three-step tool chain."""
        from llamagent.modules.fs_store.store import FSStore

        knowledge_dir = getattr(agent.config, "knowledge_dir", None)
        if not knowledge_dir:
            knowledge_dir = os.path.join(
                getattr(agent.config, "fs_data_dir", "data/fs"), "knowledge"
            )
        self._fs_store = FSStore(knowledge_dir)
        self._register_fs_tools()

    def _build_pipeline(self, agent):
        """
        Build the RetrievalPipeline via factory.

        Returns None if required backends are not installed (graceful degradation).
        Uses full hybrid retrieval (vector + lexical + optional reranker).
        """
        try:
            from llamagent.modules.rag.factory import create_pipeline
        except ImportError:
            logger.warning(
                "[Retrieval] RAG backend not available. "
                "Ensure llamagent.modules.rag is properly installed."
            )
            return None

        cfg = agent.config
        try:
            return create_pipeline(
                config=cfg,
                collection_name="llamagent_docs",
                enable_lexical=True,
                lexical_name="rag_fts",
                llm=self.llm,
                enable_reranker=getattr(cfg, "rag_rerank_enabled", False),
            )
        except Exception as e:
            logger.warning("[Retrieval] Failed to build retrieval pipeline: %s", e)
            return None

    def _register_tools(self):
        """Register search_knowledge to the Agent (tier=default)."""
        self.agent.register_tool(
            name="search_knowledge",
            func=self._tool_search_knowledge,
            description=(
                "Search relevant documents in the local knowledge base. "
                "Use cases: when the user's question involves specialized knowledge, "
                "needs reference materials, or requires querying loaded document content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords or question",
                    },
                },
                "required": ["query"],
            },
            tier="default",
            safety_level=1,
        )

    def _register_fs_tools(self):
        """Register FS backend tools: list_knowledge, list_entries, read_entry."""
        self.agent.register_tool(
            name="list_knowledge",
            func=self._tool_list_knowledge,
            description=(
                "List all available documents in the knowledge base with their "
                "title, description, and tags from frontmatter."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            tier="default",
            safety_level=1,
        )
        self.agent.register_tool(
            name="list_entries",
            func=self._tool_list_entries,
            description=(
                "List all section headings (## entries) within a specific document. "
                "Use this after list_knowledge to explore a document's structure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Document filename (e.g. 'python_best_practices.md')",
                    },
                },
                "required": ["document"],
            },
            tier="default",
            safety_level=1,
        )
        self.agent.register_tool(
            name="read_entry",
            func=self._tool_read_entry,
            description=(
                "Read the full content of a specific section within a document. "
                "Use this after list_entries to read a particular section."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Document filename (e.g. 'python_best_practices.md')",
                    },
                    "entry": {
                        "type": "string",
                        "description": "Section title to read (e.g. 'Error Handling')",
                    },
                },
                "required": ["document", "entry"],
            },
            tier="default",
            safety_level=1,
        )

    # ============================================================
    # RAG tool implementation
    # ============================================================

    def _tool_search_knowledge(self, query: str) -> str:
        """Actual execution logic for the search_knowledge tool."""
        if self.retriever is None:
            return "Knowledge base has not been initialized."

        try:
            results = self.retriever.search(query)
        except Exception as e:
            return f"Knowledge base search failed: {e}"

        if not results:
            return "No relevant content found in the knowledge base."

        # Format search results with source and relevance score
        lines = [f"Found {len(results)} relevant entries:"]
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            score = r.get("score", 0)
            content = r.get("content", "")
            lines.append(f"[{i}] (source: {source}, relevance: {score:.2f})\n{content[:500]}")
        return "\n\n".join(lines)

    # ============================================================
    # FS tool implementations
    # ============================================================

    def _tool_list_knowledge(self) -> str:
        """List all documents in the knowledge directory with their frontmatter."""
        from llamagent.modules.fs_store.parser import parse_frontmatter

        if self._fs_store is None:
            return "Knowledge base has not been initialized."

        files = self._fs_store.list_files(".md")
        if not files:
            return "No documents found in the knowledge base."

        lines = [f"Found {len(files)} document(s):"]
        for filename in files:
            content = self._fs_store.read_file(filename)
            if content is None:
                continue
            metadata, _ = parse_frontmatter(content)
            title = metadata.get("title", filename)
            description = metadata.get("description", "")
            tags = metadata.get("tags", [])
            if isinstance(tags, list):
                tags_str = ", ".join(str(t) for t in tags)
            else:
                tags_str = str(tags)

            entry = f"- **{title}** ({filename})"
            if description:
                entry += f"\n  {description}"
            if tags_str:
                entry += f"\n  Tags: {tags_str}"
            lines.append(entry)

        return "\n".join(lines)

    def _tool_list_entries(self, document: str) -> str:
        """List all section headings within a specific document."""
        from llamagent.modules.fs_store.parser import parse_frontmatter, parse_sections

        if self._fs_store is None:
            return "Knowledge base has not been initialized."

        content = self._fs_store.read_file(document)
        if content is None:
            return f"Document '{document}' not found in the knowledge base."

        metadata, body = parse_frontmatter(content)
        sections = parse_sections(body)

        if not sections:
            return f"No sections (## headings) found in '{document}'."

        title = metadata.get("title", document)
        lines = [f"Sections in '{title}' ({document}):"]
        for i, sec in enumerate(sections, 1):
            lines.append(f"  {i}. {sec['title']}")

        return "\n".join(lines)

    def _tool_read_entry(self, document: str, entry: str) -> str:
        """Read the full content of a specific section within a document."""
        from llamagent.modules.fs_store.parser import parse_frontmatter, parse_sections

        if self._fs_store is None:
            return "Knowledge base has not been initialized."

        content = self._fs_store.read_file(document)
        if content is None:
            return f"Document '{document}' not found in the knowledge base."

        _, body = parse_frontmatter(content)
        sections = parse_sections(body)

        # Find matching section (case-insensitive)
        entry_lower = entry.lower().strip()
        for sec in sections:
            if sec["title"].lower().strip() == entry_lower:
                return f"## {sec['title']}\n\n{sec['content']}"

        # Partial match fallback
        for sec in sections:
            if entry_lower in sec["title"].lower():
                return f"## {sec['title']}\n\n{sec['content']}"

        available = ", ".join(sec["title"] for sec in sections)
        return (
            f"Section '{entry}' not found in '{document}'. "
            f"Available sections: {available}"
        )

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_context(self, query: str, context: str) -> str:
        """
        Inject knowledge base usage guide based on backend.

        RAG: directs the model to use search_knowledge tool.
        FS: directs the model to use the three-step browsing tool chain.
        """
        if self._backend == "fs":
            if self._fs_store is None:
                return context
            guide = FS_RETRIEVE_GUIDE
        else:
            if self.retriever is None:
                return context
            guide = RAG_GUIDE
        return f"{context}\n\n{guide}" if context else guide

    # ------------------------------------------------------------------
    # Document loading interface (programmatic use)
    # ------------------------------------------------------------------

    def load_document(self, path: str) -> int:
        """Load a single document into the knowledge base. Returns the number of chunks."""
        if self._backend == "fs":
            logger.info(
                "[Retrieval] FS backend does not require document loading. "
                "Place .md files directly in the knowledge directory."
            )
            return 0
        if self.retriever is None:
            logger.warning("[Retrieval] Module has not been initialized")
            return 0
        return self.retriever.load_document(path)

    def load_directory(self, path: str) -> int:
        """Batch load documents from a directory into the knowledge base. Returns the total number of chunks."""
        if self._backend == "fs":
            logger.info(
                "[Retrieval] FS backend does not require document loading. "
                "Place .md files directly in the knowledge directory."
            )
            return 0
        if self.retriever is None:
            logger.warning("[Retrieval] Module has not been initialized")
            return 0
        return self.retriever.load_directory(path)

    def load_documents(self, path: str) -> int:
        """Alias for load_directory (compatibility with web_ui and api_server)."""
        return self.load_directory(path)

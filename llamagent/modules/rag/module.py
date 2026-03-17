"""
RAG Module: provides knowledge retrieval augmentation based on private documents for SmartAgent.

Tool-based (Agentic RAG) mode:
- on_attach registers the search_knowledge tool; the model autonomously decides when to retrieve
- on_context only injects a knowledge base usage guide (guiding the model to use the tool, not auto-retrieving)
- Graceful degradation when chromadb is not installed
"""

from llamagent.core.agent import Module
from llamagent.modules.rag.retriever import RAGRetriever

# Knowledge base usage guide injected into context, letting the model know it has retrieval capabilities
RAG_GUIDE = """\
[Knowledge Base] You can search the local knowledge base for professional materials.
- Use the search_knowledge tool to search relevant documents (when the user asks about specialized knowledge or needs reference materials)
- Search results may contain multiple entries; filter for the most relevant ones
- For common knowledge questions, answer directly without searching the knowledge base"""


class RAGModule(Module):
    """
    RAG Knowledge Retrieval Module.

    Tool-based integration (Agentic RAG):
    - on_attach: initialize RAGRetriever + register search_knowledge tool
    - on_context: inject knowledge base usage guide (no longer auto-retrieves)
    """

    name: str = "rag"
    description: str = "RAG knowledge retrieval: answer questions based on private documents"

    def __init__(self):
        self.retriever: RAGRetriever | None = None

    def on_attach(self, agent):
        """Initialize the retriever + register search_knowledge tool."""
        super().on_attach(agent)
        cfg = agent.config
        self.retriever = RAGRetriever(
            persist_dir=cfg.chroma_dir,
            top_k=cfg.rag_top_k,
            chunk_size=cfg.chunk_size,
        )

        # Register search_knowledge tool
        self._register_tools()

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

    # ============================================================
    # Tool implementation
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
    # Pipeline Hook
    # ============================================================

    def on_context(self, query: str, context: str) -> str:
        """
        Inject knowledge base usage guide, directing the model to retrieve via the search_knowledge tool.

        No longer auto-retrieves, consistent with the Memory module's tool-based integration approach.
        """
        if self.retriever is None:
            return context
        return f"{context}\n\n{RAG_GUIDE}" if context else RAG_GUIDE

    # ------------------------------------------------------------------
    # Document loading interface (programmatic use)
    # ------------------------------------------------------------------

    def load_document(self, path: str) -> int:
        """Load a single document into the knowledge base. Returns the number of chunks."""
        if self.retriever is None:
            print("[RAG] Module has not been initialized")
            return 0
        return self.retriever.load_document(path)

    def load_directory(self, path: str) -> int:
        """Batch load documents from a directory into the knowledge base. Returns the total number of chunks."""
        if self.retriever is None:
            print("[RAG] Module has not been initialized")
            return 0
        return self.retriever.load_directory(path)

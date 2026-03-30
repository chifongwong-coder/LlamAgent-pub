"""
RAG Retriever: document loading, chunking, and retrieval via RetrievalPipeline.

Responsibilities:
- Load documents (.txt, .md, .py, .js, .ts, .java, .go, .rs) with format-aware chunking
- Use MD5 hashing on chunk content for deduplication
- Delegate search to the shared RetrievalPipeline (vector / lexical / hybrid)
- Graceful degradation when chromadb is not installed
"""

import hashlib
import logging
import os

from llamagent.modules.rag.chunker import DocumentChunker

logger = logging.getLogger(__name__)

# Default file extensions supported for directory loading
DEFAULT_EXTENSIONS = [".txt", ".md", ".py", ".js", ".ts", ".java", ".go", ".rs"]


class RAGRetriever:
    """
    RAG Retriever: document loading, indexing, and search via RetrievalPipeline.

    Uses the shared retrieval infrastructure (RetrievalPipeline) for storage and
    search, and DocumentChunker for format-aware document splitting.

    The retrieval mode (vector / lexical / hybrid) is controlled by the _mode
    attribute, set from config during initialization.
    """

    def __init__(self, pipeline, chunker: DocumentChunker, top_k: int = 3,
                 mode: str = "hybrid"):
        """
        Initialize RAGRetriever.

        Args:
            pipeline: RetrievalPipeline instance for storage and search.
                      May be None if chromadb is unavailable (graceful degradation).
            chunker: DocumentChunker for format-aware document splitting.
            top_k: Default number of results to return from search.
            mode: Retrieval mode — "vector", "lexical", or "hybrid".
        """
        self.pipeline = pipeline
        self.chunker = chunker
        self.top_k = top_k
        self._mode = mode

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_document(self, filepath: str) -> int:
        """
        Load a single file into the knowledge base.

        Supports all P0 formats: .txt, .md, .py, .js, .ts, .java, .go, .rs.
        Uses DocumentChunker for format-aware splitting and MD5 content hashing
        for deduplication. Returns the number of chunks saved.

        Returns 0 when the pipeline is unavailable or the file does not exist.
        """
        if self.pipeline is None:
            logger.warning("[RAG] Pipeline not available, cannot load document")
            return 0

        if not os.path.exists(filepath):
            logger.warning("[RAG] File not found: %s", filepath)
            return 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning("[RAG] Failed to read file: %s - %s", filepath, e)
            return 0

        if not content.strip():
            logger.warning("[RAG] File is empty: %s", filepath)
            return 0

        chunks = self.chunker.chunk(content, filepath)
        filename = os.path.basename(filepath)

        saved = 0
        for chunk in chunks:
            # Use MD5(chunk text) as ID for deduplication
            chunk_id = hashlib.md5(chunk.text.encode("utf-8")).hexdigest()
            metadata = {
                "source": filename,
                "filepath": filepath,
                **chunk.metadata,
            }
            try:
                self.pipeline.save(chunk_id, chunk.text, metadata)
                saved += 1
            except Exception as e:
                logger.warning("[RAG] Failed to save chunk %d of %s: %s",
                               chunk.metadata.get("chunk_index", -1), filepath, e)

        logger.info("[RAG] Loaded '%s', %d chunks total", filename, saved)
        return saved

    def load_directory(self, dirpath: str, extensions: list[str] | None = None) -> int:
        """
        Load all matching files from a directory into the knowledge base.

        Args:
            dirpath: Path to the directory to scan.
            extensions: List of file extensions to include (e.g. [".txt", ".md"]).
                        Defaults to DEFAULT_EXTENSIONS.

        Returns:
            Total number of chunks saved across all files.
        """
        extensions = extensions or DEFAULT_EXTENSIONS

        if not os.path.isdir(dirpath):
            logger.warning("[RAG] Directory not found: %s", dirpath)
            return 0

        total = 0
        for name in sorted(os.listdir(dirpath)):
            if os.path.splitext(name)[1].lower() in extensions:
                total += self.load_document(os.path.join(dirpath, name))

        logger.info("[RAG] Directory loading complete, %d chunks total", total)
        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search the knowledge base using the retrieval pipeline.

        Returns:
            List of {"content": str, "source": str, "score": float}.
            Returns an empty list when the pipeline is unavailable or no results found.
        """
        if self.pipeline is None:
            return []

        k = top_k or self.top_k

        try:
            results = self.pipeline.search(query, k, mode=self._mode)
        except Exception as e:
            logger.warning("[RAG] Search failed: %s", e)
            return []

        # Normalize result format to the standard RAG output
        items = []
        for r in results:
            items.append({
                "content": r.get("text", r.get("content", "")),
                "source": r.get("metadata", {}).get("source", r.get("source", "unknown")),
                "score": r.get("score", 0.0),
            })

        return items

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return statistics about the RAG system."""
        if self.pipeline is None:
            return {"available": False}

        try:
            return {
                "available": True,
                "retrieval_mode": self._mode,
                "top_k": self.top_k,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

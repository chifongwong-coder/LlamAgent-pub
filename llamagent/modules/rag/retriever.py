"""
RAG Retriever: document vector retrieval based on ChromaDB.

Responsibilities:
- Load text files (.txt / .md) and split them into chunks by paragraph
- Use MD5 hashing on chunk content for deduplication (upsert ensures idempotency)
- Perform semantic retrieval via ChromaDB
- Lazy initialization; graceful degradation when chromadb is not installed
"""

import os
import hashlib


class RAGRetriever:
    """
    RAG Retriever: document loading, indexing, and semantic search.

    chromadb is an optional dependency with graceful degradation when not installed:
    - Lazy initialization (imports chromadb only on first call)
    - On ImportError, marks _available=False; subsequent calls silently return empty results
    - Relevance formula: round(1 / (1 + distance), 4), range (0, 1]
    """

    def __init__(self, persist_dir: str, top_k: int = 3, chunk_size: int = 500):
        self.persist_dir = persist_dir
        self.top_k = top_k
        self.chunk_size = chunk_size
        self._client = None
        self._collection = None
        self._available = True  # Whether chromadb is available

    # ------------------------------------------------------------------
    # ChromaDB lazy initialization
    # ------------------------------------------------------------------

    def _ensure_client(self) -> bool:
        """Lazily initialize the ChromaDB client. Returns True if available."""
        if self._client is not None:
            return True
        if not self._available:
            return False

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="llamagent_docs"
            )
            return True
        except ImportError:
            print("[RAG] chromadb is not installed, RAG functionality is unavailable. Please run: pip install chromadb")
            self._available = False
            return False
        except Exception as e:
            print(f"[RAG] ChromaDB initialization failed: {e}")
            self._available = False
            return False

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------

    def load_document(self, filepath: str) -> int:
        """
        Load a single file into the vector database.

        Supports .txt / .md files with MD5 deduplication. Returns the number of chunks actually written.
        Returns 0 when chromadb is unavailable or the file does not exist.
        """
        if not self._ensure_client():
            return 0

        if not os.path.exists(filepath):
            print(f"[RAG] File not found: {filepath}")
            return 0

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[RAG] Failed to read file: {filepath} - {e}")
            return 0

        if not content.strip():
            print(f"[RAG] File is empty: {filepath}")
            return 0

        chunks = self._split_text(content, self.chunk_size)
        filename = os.path.basename(filepath)

        # Use MD5(content) as chunk ID for natural deduplication
        ids, documents, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(chunk.encode("utf-8")).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "filepath": filepath,
                "chunk_index": i,
            })

        try:
            # upsert ensures idempotency
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"[RAG] Loaded '{filename}', {len(chunks)} chunks total")
        except Exception as e:
            print(f"[RAG] Failed to write document: {filepath} - {e}")
            return 0

        return len(chunks)

    def load_directory(self, dirpath: str, extensions: list[str] | None = None) -> int:
        """Batch load text files from a directory. Returns the total number of chunks."""
        extensions = extensions or [".txt", ".md"]

        if not os.path.isdir(dirpath):
            print(f"[RAG] Directory not found: {dirpath}")
            return 0

        total = 0
        for name in sorted(os.listdir(dirpath)):
            if os.path.splitext(name)[1].lower() in extensions:
                total += self.load_document(os.path.join(dirpath, name))

        print(f"[RAG] Directory loading complete, {total} chunks total")
        return total

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Semantic search, returning the most relevant document chunks.

        Returns:
            [{"content": str, "source": str, "score": float}, ...]
            Returns an empty list when chromadb is unavailable or no results are found.
        """
        if not self._ensure_client():
            return []

        top_k = top_k or self.top_k

        try:
            count = self._collection.count()
            if count == 0:
                return []

            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
            )
        except Exception as e:
            print(f"[RAG] Search failed: {e}")
            return []

        items = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                dist = results["distances"][0][i] if results["distances"] else 0
                items.append({
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    # General formula converting distance to similarity, applicable to any distance metric
                    # L2 distance range [0, +inf); this formula ensures similarity is in the range (0, 1]
                    "score": round(1 / (1 + dist), 4),
                })

        return items

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_text(text: str, chunk_size: int = 500) -> list[str]:
        """
        Split text with paragraph priority.

        Strategy: use double newlines as paragraph boundaries -> merge up to chunk_size -> force-split oversized paragraphs at chunk_size.
        """
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}" if current else para
            else:
                if current:
                    chunks.append(current)
                # Force-split oversized paragraphs
                if len(para) > chunk_size:
                    for j in range(0, len(para), chunk_size):
                        chunks.append(para[j: j + chunk_size])
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)

        return chunks

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return statistics about the RAG system."""
        if not self._ensure_client():
            return {"available": False}

        try:
            return {
                "available": True,
                "collection_name": "llamagent_docs",
                "document_count": self._collection.count(),
                "persist_dir": self.persist_dir,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

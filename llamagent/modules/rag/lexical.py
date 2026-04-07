"""Lexical (keyword) search backends for the RAG backend layer."""

import json
import sqlite3
from abc import ABC, abstractmethod


class LexicalBackend(ABC):
    """Abstract base class for lexical (keyword-based) search."""

    @abstractmethod
    def index(self, id: str, text: str, metadata: dict) -> None:
        """Index a document for keyword search."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict]:
        """Search by keyword. Returns list of {id, text, metadata, score}."""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        """Remove a document from the index."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the index."""
        ...


class SQLiteFTSBackend(LexicalBackend):
    """SQLite FTS5-based lexical search. Zero external dependencies, persistent."""

    def __init__(self, db_path: str):
        """
        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        # Create FTS5 virtual table if not exists
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(id, text, metadata)"
        )
        # Also create a regular table for id-based lookups and metadata storage
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents "
            "(id TEXT PRIMARY KEY, text TEXT, metadata TEXT)"
        )
        self._conn.commit()

    def index(self, id: str, text: str, metadata: dict) -> None:
        """Index a document. Upserts by deleting any existing entry with the same ID first."""
        meta_str = json.dumps(metadata, ensure_ascii=False)
        # Upsert: delete old then insert
        self._conn.execute("DELETE FROM fts_index WHERE id = ?", (id,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (id,))
        self._conn.execute(
            "INSERT INTO fts_index (id, text, metadata) VALUES (?, ?, ?)",
            (id, text, meta_str),
        )
        self._conn.execute(
            "INSERT INTO documents (id, text, metadata) VALUES (?, ?, ?)",
            (id, text, meta_str),
        )
        self._conn.commit()

    def search(self, query: str, top_k: int) -> list[dict]:
        """Search using FTS5 full-text matching. Returns results ranked by relevance."""
        # Escape special FTS5 characters
        safe_query = query.replace('"', '""')
        try:
            rows = self._conn.execute(
                "SELECT id, text, metadata, rank FROM fts_index "
                "WHERE fts_index MATCH ? ORDER BY rank LIMIT ?",
                (f'"{safe_query}"', top_k),
            ).fetchall()
        except Exception:
            # If FTS match fails (e.g., empty query), return empty
            return []

        results = []
        for row in rows:
            try:
                meta = json.loads(row[2]) if row[2] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            results.append({
                "id": row[0],
                "text": row[1],
                "metadata": meta,
                "score": -row[3] if row[3] else 0,  # FTS5 rank is negative (lower = better)
            })
        return results

    def delete(self, id: str) -> None:
        """Remove a document from both the FTS index and the documents table."""
        self._conn.execute("DELETE FROM fts_index WHERE id = ?", (id,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (id,))
        self._conn.commit()

    def clear(self) -> None:
        """Remove all documents from the index."""
        self._conn.execute("DELETE FROM fts_index")
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

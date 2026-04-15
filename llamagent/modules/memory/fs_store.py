"""
FSMemoryStore: file-system based memory storage using a single memories.md file.

Provides the same interface as MemoryStore so MemoryModule can use either backend.
Each memory fact is stored as a markdown section with structured metadata fields
and a blockquote source text, separated by horizontal rules.

File format:
    ## {kind}: {subject}.{attribute}
    - **fact_id**: ...
    - **kind**: ...
    - **subject**: ...
    - **attribute**: ...
    - **value**: ...
    - **confidence**: ...
    - **status**: active
    - **strength**: ...
    - **created_at**: ...
    - **updated_at**: ...

    > source_text here

    ---
"""

import logging
import re
from datetime import datetime

from llamagent.modules.fs_store.store import FSStore
from llamagent.modules.memory.fact import MemoryFact

logger = logging.getLogger(__name__)

_MEMORIES_FILE = "memories.md"

# Regex to extract metadata fields: - **key**: value
_META_RE = re.compile(r"^-\s+\*\*(\w+)\*\*:\s*(.*)$")


def _parse_sections(content: str) -> list[dict]:
    """Parse memories.md into a list of section dicts.

    Each section dict has:
        - heading: the raw ## heading text
        - meta: dict of key -> value from ``- **key**: value`` lines
        - source_text: text from ``> `` prefixed lines
        - raw: the full section text (for rewriting)
    """
    if not content or not content.strip():
        return []

    # Split by ## headings (each fact starts with ## kind: subject.attribute)
    # This is more robust than splitting on --- which could appear in source_text.
    sections = []
    current_heading = ""
    current_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        # Top-level heading (# LlamAgent Memories) — skip
        if stripped.startswith("# ") and not stripped.startswith("## "):
            continue

        # New section starts at ## heading
        if stripped.startswith("## "):
            # Save previous section
            if current_heading:
                sec = _parse_single_section(current_heading, current_lines)
                if sec:
                    sections.append(sec)
            current_heading = stripped[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Save last section
    if current_heading:
        sec = _parse_single_section(current_heading, current_lines)
        if sec:
            sections.append(sec)

    return sections


def _parse_single_section(heading: str, lines: list[str]) -> dict | None:
    """Parse a single memory section from its heading and content lines."""
    meta: dict = {}
    source_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Skip --- separators (visual decoration only)
        if stripped == "---":
            continue

        # Metadata line: - **key**: value
        m = _META_RE.match(stripped)
        if m:
            meta[m.group(1)] = m.group(2).strip()
            continue

        # Source text line: > text
        if stripped.startswith("> "):
            source_lines.append(stripped[2:])
        elif stripped == ">":
            source_lines.append("")

    if not meta:
        return None

    return {
        "heading": heading,
        "meta": meta,
        "source_text": "\n".join(source_lines),
    }


def _render_section(meta: dict, source_text: str = "") -> str:
    """Render a single memory section as markdown text.

    Args:
        meta: Dictionary with all metadata fields.
        source_text: The original source text for the memory.

    Returns:
        Formatted markdown section string.
    """
    kind = meta.get("kind", "")
    subject = meta.get("subject", "")
    attribute = meta.get("attribute", "")
    heading = f"## {kind}: {subject}.{attribute}"

    lines = [heading]

    # Metadata fields in display order
    field_order = [
        "fact_id", "kind", "subject", "attribute", "value",
        "confidence", "status", "strength",
        "created_at", "updated_at", "last_accessed_at",
    ]
    for key in field_order:
        if key in meta:
            lines.append(f"- **{key}**: {meta[key]}")

    # Source text as blockquote
    if source_text:
        lines.append("")
        for src_line in source_text.splitlines():
            lines.append(f"> {src_line}" if src_line else ">")

    return "\n".join(lines)


def _render_file(sections: list[dict]) -> str:
    """Render all sections into a complete memories.md file."""
    if not sections:
        return ""
    rendered = []
    for sec in sections:
        rendered.append(_render_section(sec["meta"], sec["source_text"]))
    return "\n\n---\n\n".join(rendered) + "\n\n---\n"


class FSMemoryStore:
    """File-system based memory store using a single memories.md file.

    Provides the same interface as MemoryStore so MemoryModule can use
    either backend transparently.

    Args:
        base_dir: Directory for the memories.md file.
    """

    def __init__(self, base_dir: str):
        self._fs = FSStore(base_dir)

    @property
    def available(self) -> bool:
        """FS store is always available (no external dependencies)."""
        return True

    # ============================================================
    # Fact-level operations
    # ============================================================

    def save_fact(self, fact: MemoryFact) -> None:
        """Append a new fact section to memories.md."""
        # Sanitize newlines in text fields to prevent metadata line corruption
        def _clean(text: str) -> str:
            return text.replace("\n", " ") if text else text

        meta = {
            "fact_id": fact.fact_id,
            "kind": fact.kind,
            "subject": _clean(fact.subject),
            "attribute": _clean(fact.attribute),
            "value": _clean(fact.value),
            "confidence": str(fact.confidence),
            "status": fact.status,
            "strength": str(fact.strength),
            "created_at": fact.created_at,
            "updated_at": fact.updated_at,
        }
        if fact.last_accessed_at:
            meta["last_accessed_at"] = fact.last_accessed_at

        section_text = _render_section(meta, fact.source_text)

        # Append to existing file
        existing = self._fs.read_file(_MEMORIES_FILE)
        if existing and existing.strip():
            new_content = existing.rstrip() + "\n\n---\n\n" + section_text + "\n\n---\n"
        else:
            new_content = section_text + "\n\n---\n"

        self._fs.write_file(_MEMORIES_FILE, new_content)

    def get_facts_by_key(
        self, kind: str, subject: str, attribute: str
    ) -> list[dict]:
        """Find sections matching (kind, subject, attribute) exactly.

        Returns a list of metadata dicts (same format as MemoryStore returns).
        """
        sections = self._read_sections()
        matches = []
        for sec in sections:
            meta = sec["meta"]
            if (
                meta.get("kind") == kind
                and meta.get("subject") == subject
                and meta.get("attribute") == attribute
            ):
                matches.append(meta)
        return matches

    def update_fact_status(self, fact_id: str, status: str) -> None:
        """Update the status field of a fact identified by fact_id."""
        sections = self._read_sections()
        modified = False
        for sec in sections:
            if sec["meta"].get("fact_id") == fact_id:
                sec["meta"]["status"] = status
                sec["meta"]["updated_at"] = datetime.now().isoformat()
                modified = True
                break

        if modified:
            self._write_sections(sections)

    def update_fact_accessed(
        self,
        fact_id: str,
        increment_strength: float = 0.1,
        max_strength: float = 2.0,
    ) -> None:
        """Update last_accessed_at and reinforce strength for a fact."""
        sections = self._read_sections()
        modified = False
        for sec in sections:
            if sec["meta"].get("fact_id") == fact_id:
                sec["meta"]["last_accessed_at"] = datetime.now().isoformat()
                current_strength = float(sec["meta"].get("strength", "1.0"))
                new_strength = min(current_strength + increment_strength, max_strength)
                sec["meta"]["strength"] = str(round(new_strength, 2))
                modified = True
                break

        if modified:
            self._write_sections(sections)

    def search_facts(
        self, query: str, top_k: int = 5, status_filter: str = "active"
    ) -> list[dict]:
        """Search facts by keyword matching (FS has no vector search).

        Returns results in the same format as MemoryStore.search_facts for
        compatibility with the scoring pipeline.
        """
        sections = self._read_sections()
        results = []
        query_lower = query.lower()
        query_terms = query_lower.split()

        for sec in sections:
            meta = sec["meta"]
            if status_filter and meta.get("status", "active") != status_filter:
                continue

            # Build searchable text from all metadata + source
            searchable = " ".join([
                meta.get("kind", ""),
                meta.get("subject", ""),
                meta.get("attribute", ""),
                meta.get("value", ""),
                sec.get("source_text", ""),
            ]).lower()

            # Simple keyword matching: count how many query terms appear
            hits = sum(1 for term in query_terms if term in searchable)
            if hits == 0:
                continue

            score = hits / max(len(query_terms), 1)
            results.append({
                "id": meta.get("fact_id", ""),
                "text": f"{meta.get('kind', '')}: {meta.get('subject', '')}.{meta.get('attribute', '')} = {meta.get('value', '')}",
                "metadata": meta,
                "score": round(score, 4),
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def clear(self) -> None:
        """Clear all memories by deleting the file."""
        self._fs.delete_file(_MEMORIES_FILE)

    def get_stats(self) -> dict:
        """Return memory statistics."""
        sections = self._read_sections()
        return {"available": True, "count": len(sections)}

    # ============================================================
    # FS-specific methods (used by MemoryModule FS backend)
    # ============================================================

    def list_all_active_facts(self) -> list[dict]:
        """Return metadata dicts for all active facts.

        Returns:
            List of metadata dicts (each dict has fact_id, kind, subject, etc.).
        """
        sections = self._read_sections()
        return [s["meta"] for s in sections if s["meta"].get("status", "active") == "active"]

    def update_fact_value(self, fact_id: str, new_value: str) -> None:
        """Update a fact's value and updated_at timestamp.

        Args:
            fact_id: The fact_id to update.
            new_value: New value string for the fact.
        """
        sections = self._read_sections()
        modified = False
        for sec in sections:
            if sec["meta"].get("fact_id") == fact_id:
                sec["meta"]["value"] = new_value
                sec["meta"]["updated_at"] = datetime.now().isoformat()
                modified = True
                break

        if modified:
            self._write_sections(sections)

    def list_all_metadata(self) -> str:
        """Return all active facts as formatted metadata text for context injection.

        Format: one line per fact with fact_id, kind, subject.attribute, value.
        """
        sections = self._read_sections()
        lines = []
        for sec in sections:
            meta = sec["meta"]
            if meta.get("status", "active") != "active":
                continue
            fact_id = meta.get("fact_id", "?")
            kind = meta.get("kind", "?")
            subject = meta.get("subject", "?")
            attribute = meta.get("attribute", "?")
            value = meta.get("value", "")
            lines.append(f"- [{fact_id}] {kind}: {subject}.{attribute} = {value}")

        if not lines:
            return ""
        return "\n".join(lines)

    def read_fact_source(self, fact_id: str) -> str:
        """Return metadata summary + source_text for a specific fact."""
        sections = self._read_sections()
        for sec in sections:
            meta = sec["meta"]
            if meta.get("fact_id") == fact_id:
                parts = []
                parts.append(f"Kind: {meta.get('kind', '?')}")
                parts.append(f"Subject: {meta.get('subject', '?')}")
                parts.append(f"Attribute: {meta.get('attribute', '?')}")
                parts.append(f"Value: {meta.get('value', '?')}")
                parts.append(f"Confidence: {meta.get('confidence', '?')}")
                parts.append(f"Status: {meta.get('status', '?')}")
                parts.append(f"Strength: {meta.get('strength', '?')}")
                parts.append(f"Created: {meta.get('created_at', '?')}")
                parts.append(f"Updated: {meta.get('updated_at', '?')}")
                source = sec.get("source_text", "")
                if source:
                    parts.append(f"\nSource:\n{source}")
                return "\n".join(parts)
        return f"Memory with fact_id '{fact_id}' not found."

    # ============================================================
    # Legacy interface (backward compatibility with MemoryStore)
    # ============================================================

    def save_memory(self, content: str, category: str = "conversation") -> None:
        """Save content as a plain text fact (legacy interface)."""
        import uuid
        now = datetime.now().isoformat()
        fact = MemoryFact(
            fact_id=uuid.uuid4().hex,
            kind=category if category != "conversation" else "episode",
            subject="user",
            attribute="note",
            value=content[:200],
            source_text=content,
            created_at=now,
            updated_at=now,
        )
        self.save_fact(fact)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword search on all active facts (legacy interface)."""
        results = self.search_facts(query, top_k=top_k, status_filter="active")
        items = []
        for r in results:
            meta = r.get("metadata", {})
            items.append({
                "content": meta.get("value", r.get("text", "")),
                "category": meta.get("kind", ""),
                "created_at": meta.get("created_at", ""),
                "score": r.get("score", 0.0),
            })
        return items

    def clear_long_term(self) -> None:
        """Clear all long-term memories (legacy alias for clear())."""
        self.clear()

    # ============================================================
    # Internal helpers
    # ============================================================

    def _read_sections(self) -> list[dict]:
        """Read and parse all sections from memories.md."""
        content = self._fs.read_file(_MEMORIES_FILE)
        if not content:
            return []
        return _parse_sections(content)

    def _write_sections(self, sections: list[dict]) -> None:
        """Rewrite the entire memories.md from parsed sections."""
        content = _render_file(sections)
        self._fs.write_file(_MEMORIES_FILE, content)

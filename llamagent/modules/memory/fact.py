"""
MemoryFact data model and CompileResult container.

A MemoryFact represents a single structured piece of information extracted from
conversation. Facts are stored in vector storage with metadata for efficient
retrieval and deduplication via (kind, subject, attribute) keys.
"""

from dataclasses import dataclass, field
from datetime import datetime

VALID_KINDS = ("preference", "profile", "project_fact", "instruction", "decision", "episode")


@dataclass
class MemoryFact:
    """
    A single structured memory fact.

    Attributes:
        fact_id: Unique identifier for this fact.
        kind: Category of fact; must be one of VALID_KINDS.
        subject: Normalized subject key (e.g. "user", "project_x").
        attribute: Normalized attribute key (e.g. "preferred_language").
        value: The actual value or statement.
        confidence: Extraction confidence score (0.0 to 1.0).
        source_text: Original text from which this fact was extracted.
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        last_accessed_at: ISO timestamp of last retrieval access.
        strength: Decay-adjusted strength score for ranking.
        status: Lifecycle status: "active" / "superseded" / "archived".
    """

    fact_id: str
    kind: str
    subject: str
    attribute: str
    value: str
    confidence: float = 1.0
    source_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed_at: str = ""
    strength: float = 1.0
    status: str = "active"

    def to_metadata(self) -> dict:
        """Convert to flat metadata dict for vector backend storage."""
        return {
            "fact_id": self.fact_id,
            "kind": self.kind,
            "subject": self.subject,
            "attribute": self.attribute,
            "value": self.value,
            "confidence": self.confidence,
            "source_text": self.source_text[:500],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_accessed_at": self.last_accessed_at,
            "strength": self.strength,
            "status": self.status,
        }

    def to_text(self) -> str:
        """Convert to embeddable text for vector storage."""
        return f"{self.kind}: {self.subject}.{self.attribute} = {self.value}"


@dataclass
class CompileResult:
    """Result of fact compilation from free text."""

    facts: list[MemoryFact] = field(default_factory=list)
    fallback_text: str | None = None
    success: bool = True


def normalize_key(s: str) -> str:
    """Normalize a subject or attribute string for consistent matching."""
    return s.strip().lower().replace("-", "_").replace(" ", "_")

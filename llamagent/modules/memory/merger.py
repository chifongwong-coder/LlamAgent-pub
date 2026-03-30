"""
FactMerger: deduplication and update logic for incoming MemoryFacts.

When a new fact arrives, the merger checks whether a fact with the same
(kind, subject, attribute) triple already exists. If so, it decides whether
to update the existing fact or skip the new one.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

from llamagent.modules.memory.fact import MemoryFact

logger = logging.getLogger(__name__)


@dataclass
class MergeAction:
    """
    Result of a merge decision.

    Attributes:
        action: One of "insert" / "update" / "skip".
        fact: The fact to insert or the updated fact (with merged fields).
        superseded: The old fact that was superseded (only set when action="update").
    """

    action: str  # "insert" / "update" / "skip"
    fact: MemoryFact
    superseded: MemoryFact | None = None


class FactMerger:
    """
    Handles deduplication and merging of incoming facts against existing ones.

    Matching uses exact (kind, normalized_subject, normalized_attribute) triple.
    The merger queries the MemoryStore to find existing facts with matching keys.
    """

    def __init__(self, store):
        """
        Initialize with a MemoryStore for lookups.

        Args:
            store: A MemoryStore instance with get_facts_by_key() method.
        """
        self._store = store

    def merge(self, new_fact: MemoryFact) -> MergeAction:
        """
        Decide how to handle a new fact given existing facts in the store.

        Logic:
        1. Look up existing active facts with matching (kind, subject, attribute).
        2. If no match found: action = "insert".
        3. If match found with identical value: action = "skip" (exact duplicate).
        4. If match found with different value: action = "update" (supersede old fact).

        Args:
            new_fact: The incoming MemoryFact to merge.

        Returns:
            MergeAction describing what to do.
        """
        try:
            existing = self._store.get_facts_by_key(
                kind=new_fact.kind,
                subject=new_fact.subject,
                attribute=new_fact.attribute,
            )
        except Exception as e:
            logger.warning("[Memory] Merger lookup failed: %s; defaulting to insert", e)
            return MergeAction(action="insert", fact=new_fact)

        if not existing:
            return MergeAction(action="insert", fact=new_fact)

        # Find the most recent active match
        best_match = None
        for candidate in existing:
            if candidate.get("status", "active") != "active":
                continue
            if best_match is None:
                best_match = candidate
            else:
                # Prefer the one with more recent updated_at
                if candidate.get("updated_at", "") > best_match.get("updated_at", ""):
                    best_match = candidate

        if best_match is None:
            return MergeAction(action="insert", fact=new_fact)

        old_value = best_match.get("value", "")

        # Exact duplicate — skip (return existing fact_id for reinforcement)
        if old_value.strip().lower() == new_fact.value.strip().lower():
            existing_fact = MemoryFact(
                fact_id=best_match.get("fact_id", new_fact.fact_id),
                kind=new_fact.kind,
                subject=new_fact.subject,
                attribute=new_fact.attribute,
                value=new_fact.value,
            )
            return MergeAction(action="skip", fact=existing_fact)

        # Different value — update (supersede old fact)
        now = datetime.now().isoformat()

        # Build the superseded old fact record
        old_fact = MemoryFact(
            fact_id=best_match.get("fact_id", ""),
            kind=best_match.get("kind", new_fact.kind),
            subject=best_match.get("subject", new_fact.subject),
            attribute=best_match.get("attribute", new_fact.attribute),
            value=old_value,
            confidence=float(best_match.get("confidence", 1.0)),
            source_text=best_match.get("source_text", ""),
            created_at=best_match.get("created_at", ""),
            updated_at=now,
            last_accessed_at=best_match.get("last_accessed_at", ""),
            strength=float(best_match.get("strength", 1.0)),
            status="superseded",
        )

        # The new fact inherits the old fact's created_at for continuity
        new_fact.updated_at = now
        new_fact.created_at = best_match.get("created_at", new_fact.created_at)

        return MergeAction(
            action="update",
            fact=new_fact,
            superseded=old_fact,
        )

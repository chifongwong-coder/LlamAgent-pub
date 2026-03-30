"""
FactCompiler: uses an LLM to extract structured MemoryFacts from conversation text.

Two modes of operation:
- compile(text): Extract facts from arbitrary text (used by save_memory tool).
- compile_hybrid(query, response): Combined "should_store" decision + fact extraction
  in a single LLM call (used by hybrid mode on_output).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime

from llamagent.modules.memory.fact import (
    VALID_KINDS,
    CompileResult,
    MemoryFact,
    normalize_key,
)

logger = logging.getLogger(__name__)

# Prompt for extracting structured facts from text
FACT_EXTRACTION_PROMPT = """\
Extract structured facts from the following text. Each fact should capture a single
piece of information with these fields:
- kind: one of (preference, profile, project_fact, instruction, decision, episode)
- subject: the entity this fact is about (e.g. "user", "project_x"), in snake_case
- attribute: the specific attribute (e.g. "preferred_language", "deadline"), in snake_case
- value: the actual value or statement

Return a JSON array of fact objects. If no meaningful facts can be extracted, return an empty array [].

Example output:
[
  {"kind": "preference", "subject": "user", "attribute": "response_style", "value": "concise"},
  {"kind": "project_fact", "subject": "llamagent", "attribute": "tech_stack", "value": "Python + LiteLLM + ChromaDB"}
]

Text to extract from:
"""

# Prompt for hybrid mode: combined store decision + fact extraction
HYBRID_PROMPT = """\
Analyze the following conversation turn and determine:
1. Whether it contains information worth remembering long-term (user preferences,
   important facts, key conclusions, work information, decisions, instructions).
2. If so, extract structured facts from it.

Return JSON with this structure:
{
  "should_store": true/false,
  "summary": "One or two sentences summarizing the key information (empty string if should_store=false)",
  "facts": [
    {"kind": "preference", "subject": "user", "attribute": "...", "value": "..."},
    ...
  ]
}

Rules:
- should_store=false for casual chat, greetings, or content without substance
- kind must be one of: preference, profile, project_fact, instruction, decision, episode
- subject and attribute should be in snake_case
- facts array should be empty if should_store=false

Conversation:
"""


@dataclass
class HybridResult:
    """Result of hybrid compilation (store decision + fact extraction)."""

    should_store: bool = False
    facts: list[MemoryFact] = field(default_factory=list)
    summary: str = ""
    success: bool = True


class FactCompiler:
    """
    Extracts structured MemoryFacts from text using an LLM.

    The compiler sends text to the LLM with a structured extraction prompt,
    parses the JSON response, validates kinds, and normalizes keys.
    """

    def __init__(self, llm_client):
        """
        Initialize with an LLM client.

        Args:
            llm_client: An LLMClient instance with ask_json() method.
        """
        self._llm = llm_client

    def compile(self, text: str) -> CompileResult:
        """
        Extract structured facts from free text.

        On JSON parse failure or invalid output, returns a CompileResult with
        fallback_text set and success=False. On success, normalizes subject/attribute
        and validates kind (invalid kind maps to "episode").

        Args:
            text: The text to extract facts from.

        Returns:
            CompileResult with extracted facts or fallback text.
        """
        try:
            result = self._llm.ask_json(
                prompt=FACT_EXTRACTION_PROMPT + text,
                system="You are a fact extraction engine. Return only valid JSON.",
            )
        except Exception as e:
            logger.warning("[Memory] Fact extraction LLM call failed: %s", e)
            return CompileResult(facts=[], fallback_text=text, success=False)

        # The LLM may return a dict with a wrapper key or a raw list
        raw_facts = result
        if isinstance(result, dict):
            # Try common wrapper keys
            for key in ("facts", "results", "data"):
                if key in result and isinstance(result[key], list):
                    raw_facts = result[key]
                    break
            else:
                # If it's a dict but has no list wrapper, treat as failure
                if not isinstance(raw_facts, list):
                    logger.warning("[Memory] Unexpected LLM output format: %s", type(result))
                    return CompileResult(facts=[], fallback_text=text, success=False)

        if not isinstance(raw_facts, list):
            logger.warning("[Memory] Expected list of facts, got %s", type(raw_facts))
            return CompileResult(facts=[], fallback_text=text, success=False)

        if len(raw_facts) == 0:
            return CompileResult(facts=[], fallback_text=text, success=False)

        facts = []
        now = datetime.now().isoformat()
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact = self._parse_fact(item, source_text=text, timestamp=now)
            if fact is not None:
                facts.append(fact)

        if not facts:
            return CompileResult(facts=[], fallback_text=text, success=False)

        return CompileResult(facts=facts, success=True)

    def compile_hybrid(self, query: str, response: str) -> HybridResult:
        """
        Combined "should_store" decision + fact extraction in a single LLM call.

        Used by hybrid mode on_output to avoid two separate LLM calls.

        Args:
            query: The user's query.
            response: The assistant's response.

        Returns:
            HybridResult with store decision, facts, and summary.
        """
        conversation_text = f"User: {query}\nAssistant: {response}"

        try:
            result = self._llm.ask_json(
                prompt=HYBRID_PROMPT + conversation_text,
                system="You are a memory analysis engine. Return only valid JSON.",
            )
        except Exception as e:
            logger.warning("[Memory] Hybrid compilation LLM call failed: %s", e)
            return HybridResult(should_store=False, success=False)

        if not isinstance(result, dict):
            logger.warning("[Memory] Hybrid result is not a dict: %s", type(result))
            return HybridResult(should_store=False, success=False)

        should_store = bool(result.get("should_store", False))
        summary = str(result.get("summary", ""))

        if not should_store:
            return HybridResult(should_store=False, summary=summary, success=True)

        # Extract facts from the result
        raw_facts = result.get("facts", [])
        if not isinstance(raw_facts, list):
            raw_facts = []

        now = datetime.now().isoformat()
        facts = []
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact = self._parse_fact(item, source_text=conversation_text, timestamp=now)
            if fact is not None:
                facts.append(fact)

        return HybridResult(
            should_store=True,
            facts=facts,
            summary=summary,
            success=True,
        )

    def _parse_fact(
        self, item: dict, source_text: str = "", timestamp: str = ""
    ) -> MemoryFact | None:
        """
        Parse a single fact dict from LLM output into a MemoryFact.

        Validates kind (invalid kind maps to "episode"), normalizes subject/attribute,
        and generates a deterministic fact_id.

        Returns None if required fields are missing.
        """
        kind = str(item.get("kind", "episode")).strip().lower()
        subject = item.get("subject", "")
        attribute = item.get("attribute", "")
        value = item.get("value", "")

        # Skip if essential fields are empty
        if not subject or not attribute or not value:
            return None

        # Normalize
        subject = normalize_key(subject)
        attribute = normalize_key(attribute)

        # Validate kind — invalid kind falls back to "episode"
        if kind not in VALID_KINDS:
            kind = "episode"

        # Generate deterministic fact_id from key triple + value
        id_source = f"{kind}:{subject}:{attribute}:{value}"
        fact_id = hashlib.md5(id_source.encode()).hexdigest()

        ts = timestamp or datetime.now().isoformat()

        return MemoryFact(
            fact_id=fact_id,
            kind=kind,
            subject=subject,
            attribute=attribute,
            value=value,
            confidence=float(item.get("confidence", 1.0)),
            source_text=source_text,
            created_at=ts,
            updated_at=ts,
            strength=1.0,
            status="active",
        )

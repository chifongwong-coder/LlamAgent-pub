"""
FactCompiler: uses an LLM to extract structured MemoryFacts from conversation text.

Compile modes:
- compile(text): Extract JSON atomic facts (kind/subject/attribute/value) — used
  when memory_compile_mode='structured' (natural for RAG/DB backends).
- compile_raw_text(text, category): Produce a single subject+body entry using
  <subject>/<body> tags — used when memory_compile_mode='raw_text' (natural for
  the FS backend where retrieval is full-dump + LLM attention).
- compile_hybrid(query, response): Combined "should_store" decision + fact
  extraction in a single LLM call (used by hybrid mode on_output).

Dedup helpers:
- judge_duplicate(new_subject, new_body, existing): LLM-based judgement for the
  raw_text path when count-threshold triggers dedup.
"""

import hashlib
import logging
import re
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

    # ==================================================================
    # Raw-text compile path (for FS backend / weak-model friendly)
    # ==================================================================

    def compile_raw_text(
        self, text: str, category: str = "episode"
    ) -> MemoryFact | None:
        """Produce a single subject+body memory entry.

        A single LLM call; the model returns ``<subject>...</subject>`` and
        ``<body>...</body>`` tags. Parsing is lenient: if the tags are absent
        or malformed, the whole reply becomes the body and subject is empty.

        The returned MemoryFact uses attribute="note" as a placeholder to keep
        the schema stable; the FS backend renders it as a markdown section.

        Args:
            text: The text to remember (as given to save_memory).
            category: User-supplied label, mapped to MemoryFact.kind; invalid
                values fall back to "episode" per VALID_KINDS.

        Returns:
            A MemoryFact with subject/body populated, or None on LLM failure.
        """
        try:
            reply = self._llm.ask(
                prompt=RAW_TEXT_COMPILE_PROMPT + text,
                system=(
                    "You produce short memory entries. Always respond with "
                    "<subject>...</subject> and <body>...</body> tags."
                ),
            )
        except Exception as e:
            logger.warning("[Memory] Raw-text compile LLM call failed: %s", e)
            return None

        subject, body = _parse_raw_text_tags(reply, fallback_body=text)
        if not body:
            body = text

        kind = (category or "").strip().lower()
        if kind not in VALID_KINDS:
            kind = "episode"

        ts = datetime.now().isoformat()
        id_source = f"{kind}:raw_text:{subject}:{body}:{ts}"
        fact_id = hashlib.md5(id_source.encode()).hexdigest()

        return MemoryFact(
            fact_id=fact_id,
            kind=kind,
            subject=subject or "entry",
            attribute="note",
            value=body,
            confidence=1.0,
            source_text=text,
            created_at=ts,
            updated_at=ts,
            strength=1.0,
            status="active",
        )

    def judge_duplicate(
        self,
        new_subject: str,
        new_body: str,
        existing: list[dict],
    ) -> str | None:
        """LLM-based duplicate judgement for raw-text entries.

        Only invoked when ``memory_dedup_threshold`` triggers. The LLM is
        shown the new entry alongside existing subjects and asked whether the
        new content is semantically equivalent to any existing one. Returns
        the matched fact_id, or None if the model says no match / the call
        fails / the reply cannot be parsed.

        Args:
            new_subject: Subject of the new entry.
            new_body: Body of the new entry.
            existing: List of metadata dicts (each with fact_id + subject).

        Returns:
            Matched fact_id or None.
        """
        if not existing:
            return None

        lines = [
            f"- id={meta.get('fact_id', '')} subject={meta.get('subject', '')}"
            for meta in existing
        ]
        existing_listing = "\n".join(lines)

        prompt = (
            "New memory entry:\n"
            f"  subject: {new_subject}\n"
            f"  body: {new_body}\n\n"
            "Existing entries:\n"
            f"{existing_listing}\n\n"
            "Does the new entry describe the same thing as one of the existing "
            "entries? If yes, reply with exactly that entry's id surrounded by "
            "<match>...</match> tags. If no existing entry matches, reply with "
            "<match>none</match>."
        )

        try:
            reply = self._llm.ask(
                prompt=prompt,
                system=(
                    "You judge whether two memory entries describe the same "
                    "thing. Respond with a single <match>...</match> tag."
                ),
            )
        except Exception as e:
            logger.warning("[Memory] Dedup judgement LLM call failed: %s", e)
            return None

        match = re.search(r"<match>\s*(.*?)\s*</match>", reply, flags=re.DOTALL)
        if not match:
            return None

        candidate = match.group(1).strip()
        if not candidate or candidate.lower() == "none":
            return None

        valid_ids = {meta.get("fact_id", "") for meta in existing}
        if candidate not in valid_ids:
            return None

        return candidate


# ======================================================================
# Raw-text compile prompt and tag parser
# ======================================================================

RAW_TEXT_COMPILE_PROMPT = """\
Summarize the following text as one short memory entry.

Respond with exactly two tagged fields:
<subject>A brief title (3-10 words) capturing what this memory is about</subject>
<body>The content to remember, preserving the important details</body>

If you cannot produce a meaningful subject, emit <subject></subject> and place
the full content in <body>.

Text:
"""


def _parse_raw_text_tags(reply: str, fallback_body: str = "") -> tuple[str, str]:
    """Parse ``<subject>...</subject>`` and ``<body>...</body>`` tags.

    Lenient: accepts missing / mismatched tags. If ``<body>`` is absent, the
    whole reply (minus any ``<subject>`` match) is treated as the body. If
    the body ends up empty, ``fallback_body`` is used.

    Returns:
        (subject, body) — subject may be empty; body is guaranteed non-empty
        if either the reply or fallback_body had content.
    """
    if not reply:
        return "", fallback_body

    subject_match = re.search(
        r"<subject>\s*(.*?)\s*</subject>", reply, flags=re.DOTALL | re.IGNORECASE
    )
    body_match = re.search(
        r"<body>\s*(.*?)\s*</body>", reply, flags=re.DOTALL | re.IGNORECASE
    )

    subject = subject_match.group(1).strip() if subject_match else ""

    if body_match:
        body = body_match.group(1).strip()
    else:
        # Strip any <subject>...</subject> segment and use the remainder
        body = re.sub(
            r"<subject>.*?</subject>",
            "",
            reply,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

    if not body:
        body = fallback_body

    return subject, body

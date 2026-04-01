"""
MemoryModule: structured long-term memory with fact extraction, merging, and auto-recall.

v1.7 rewrite: replaces the v1.0 plain-text memory with a structured fact system.

Write modes (memory_mode):
- off:         Memory disabled, no save tool registered
- autonomous:  Model calls save_memory via function calling
- hybrid:      autonomous + on_output auto-extracts facts via compile_hybrid()

Read modes (memory_recall_mode):
- off:         No recall tool registered, no auto recall
- tool:        Model calls recall_memory via function calling
- auto:        Automatic context injection on every turn (threshold-gated) + tool available

The module uses:
- FactCompiler: LLM-based extraction of structured MemoryFacts
- FactMerger: deduplication via (kind, subject, attribute) key matching
- MemoryStore: persistence via RetrievalPipeline (ChromaDB vector backend)
"""

import logging
import math
import time
from datetime import datetime

from llamagent.core.agent import Module
from llamagent.modules.memory.compiler import FactCompiler
from llamagent.modules.memory.fact import CompileResult, MemoryFact
from llamagent.modules.memory.merger import FactMerger, MergeAction
from llamagent.modules.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# Memory usage guide injected into context
MEMORY_GUIDE = """\
[Memory] You have long-term memory and can remember information across conversations.
- Use the save_memory tool to save noteworthy content (user preferences, important facts, \
key conclusions, decisions, instructions).
- Use the recall_memory tool to recall past information when relevant.
- Do not save casual chat or meaningless content; only save truly valuable information."""

# Shortened guide when only save is available (recall_mode = off)
MEMORY_GUIDE_SAVE_ONLY = """\
[Memory] You have long-term memory and can remember information across conversations.
- Use the save_memory tool to save noteworthy content (user preferences, important facts, \
key conclusions, decisions, instructions).
- Do not save casual chat or meaningless content; only save truly valuable information."""

# Shortened guide when only recall is available (memory_mode = off)
MEMORY_GUIDE_RECALL_ONLY = """\
[Memory] You have long-term memory with stored information from previous conversations.
- Use the recall_memory tool to recall past information when relevant."""

# Auto-recall context block template
_AUTO_RECALL_BLOCK = """\
[Memory] Relevant memories for this conversation:
{memories}"""

# Valid modes
_VALID_WRITE_MODES = ("off", "autonomous", "hybrid")
_VALID_READ_MODES = ("off", "tool", "auto")

# Short greeting patterns to skip in auto-recall
_GREETING_PATTERNS = frozenset({
    "hi", "hello", "hey", "yo", "sup", "good morning", "good afternoon",
    "good evening", "good night", "thanks", "thank you", "bye", "goodbye",
    "see you", "ok", "okay", "yes", "no", "sure", "yep", "nope",
})

# Kind priority bonuses for scoring (higher = more relevant by default)
_KIND_PRIORITY = {
    "instruction": 0.10,
    "preference": 0.08,
    "decision": 0.05,
    "project_fact": 0.03,
    "profile": 0.02,
    "episode": 0.00,
}


def _is_short_greeting(text: str) -> bool:
    """Check if text is a short greeting that should skip auto-recall."""
    cleaned = text.strip().lower().rstrip("!?.,:;")
    if len(cleaned) < 2:
        return True
    return cleaned in _GREETING_PATTERNS


class MemoryModule(Module):
    """
    Memory Module v1.7: structured fact extraction, merging, and auto-recall.

    Behavioral matrix:
    - memory_mode controls writing: off / autonomous / hybrid
    - memory_recall_mode controls reading: off / tool / auto
    - Tools are registered independently based on each mode
    """

    name: str = "memory"
    description: str = "Long-term memory: structured facts with extraction, merging, and auto-recall"

    def __init__(self):
        self.store: MemoryStore | None = None
        self.compiler: FactCompiler | None = None
        self.merger: FactMerger | None = None
        self._write_mode: str = "off"
        self._read_mode: str = "off"
        self._available: bool = False
        self._pending_query: str | None = None  # For hybrid mode on_output

    # ============================================================
    # Lifecycle
    # ============================================================

    def on_attach(self, agent):
        """Initialize retrieval pipeline, compiler, merger, and register tools."""
        super().on_attach(agent)

        # Resolve modes from config
        self._write_mode = getattr(agent.config, "memory_mode", "off")
        if self._write_mode not in _VALID_WRITE_MODES:
            logger.warning(
                "[Memory] Unknown memory_mode='%s', falling back to 'off'",
                self._write_mode,
            )
            self._write_mode = "off"

        self._read_mode = getattr(agent.config, "memory_recall_mode", "tool")
        if self._read_mode not in _VALID_READ_MODES:
            logger.warning(
                "[Memory] Unknown memory_recall_mode='%s', falling back to 'tool'",
                self._read_mode,
            )
            self._read_mode = "tool"

        # If both modes are off, nothing to do
        if self._write_mode == "off" and self._read_mode == "off":
            logger.info("[Memory] Both write and read modes are off; memory disabled")
            self._available = False
            return

        # Build retrieval pipeline with graceful degradation
        pipeline = self._build_pipeline(agent)

        if pipeline is not None:
            self.store = MemoryStore(pipeline=pipeline)
            self._available = True
        else:
            # Degraded: store exists but has no backend
            self.store = MemoryStore()
            self._available = False
            logger.warning(
                "[Memory] Storage unavailable (chromadb not installed?). "
                "Tools will return friendly error messages."
            )

        # Build compiler (needs LLM for fact extraction)
        if self._write_mode != "off":
            try:
                self.compiler = FactCompiler(agent.llm)
            except Exception as e:
                logger.warning("[Memory] Failed to create FactCompiler: %s", e)
                self.compiler = None

        # Build merger (needs store for key lookups)
        if self.store is not None:
            self.merger = FactMerger(self.store)

        # Register tools based on modes
        if self._write_mode != "off":
            self._register_save_tool()
        if self._read_mode != "off":
            self._register_recall_tool()

    def _build_pipeline(self, agent):
        """
        Build a RetrievalPipeline for memory storage via factory.

        Returns None on failure (e.g. chromadb not installed).
        Memory uses vector-only retrieval (no lexical/reranker).
        """
        try:
            from llamagent.modules.retrieval.factory import create_pipeline
        except ImportError:
            logger.warning("[Memory] Retrieval module not available")
            return None

        # Determine collection name (isolated by persona)
        if agent.persona:
            collection = f"memory_{agent.persona.persona_id}"
        else:
            collection = "llamagent_memory"

        try:
            return create_pipeline(
                config=agent.config,
                collection_name=collection,
                enable_lexical=False,
                enable_reranker=False,
            )
        except Exception as e:
            logger.warning("[Memory] Failed to build retrieval pipeline: %s", e)
            return None

    # ============================================================
    # Tool registration
    # ============================================================

    def _register_save_tool(self):
        """Register the save_memory tool (when memory_mode != 'off')."""
        self.agent.register_tool(
            name="save_memory",
            func=self._tool_save_memory,
            description=(
                "Save important information to long-term memory as structured facts. "
                "Use cases: user mentions personal preferences, important facts, work information, "
                "explicitly asks to remember something, or the conversation produces conclusions worth keeping."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The content to remember",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Category hint: preference / profile / project_fact / "
                            "instruction / decision / episode"
                        ),
                    },
                },
                "required": ["content"],
            },
            tier="default",
            safety_level=2,
        )

    def _register_recall_tool(self):
        """Register the recall_memory tool (when memory_recall_mode != 'off')."""
        self.agent.register_tool(
            name="recall_memory",
            func=self._tool_recall_memory,
            description=(
                "Search for relevant information from long-term memory. "
                "Use cases: user asks about something discussed before, needs to recall user preferences, "
                "or the current conversation requires historical context."
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
    # Tool implementations
    # ============================================================

    def _tool_save_memory(self, content: str, category: str = "general") -> str:
        """Save memory tool: compile text into facts, then merge each into the store."""
        if not self._available or self.store is None:
            return "Memory storage is currently unavailable. Information was not saved."

        if self.compiler is None:
            # No compiler available: fall back to plain text save
            return self._save_fallback(content, category)

        try:
            result = self.compiler.compile(content)
        except Exception as e:
            logger.warning("[Memory] Compile failed: %s", e)
            return self._save_fallback(content, category)

        if not result.success or not result.facts:
            return self._handle_compile_failure(result, content, category)

        return self._process_facts(result.facts)

    def _tool_recall_memory(self, query: str) -> str:
        """Recall memory tool: search facts with scoring bonuses."""
        if not self._available or self.store is None:
            return "Memory storage is currently unavailable."

        top_k = getattr(self.agent.config, "memory_recall_top_k", 5)

        try:
            results = self.store.search_facts(query, top_k=top_k, status_filter="active")
        except Exception as e:
            return f"Failed to retrieve memories: {e}"

        if not results:
            return "No relevant memories found."

        # Apply scoring bonuses and sort
        scored = self._apply_scoring(results)

        # Update access timestamps
        for item in scored:
            fact_id = item.get("metadata", {}).get("fact_id", item.get("id", ""))
            if fact_id:
                try:
                    self.store.update_fact_accessed(fact_id)
                except Exception:
                    pass  # Best-effort access tracking

        # Format results
        lines = ["Found the following relevant memories:"]
        for item in scored:
            meta = item.get("metadata", {})
            score = item.get("final_score", item.get("score", 0))
            kind = meta.get("kind", "")
            value = meta.get("value", item.get("text", ""))
            kind_tag = f"[{kind}] " if kind else ""
            lines.append(f"- (relevance {score:.3f}) {kind_tag}{value}")

        return "\n".join(lines)

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_context(self, query: str, context: str) -> str:
        """
        Inject memory guide and perform auto-recall if enabled.

        - Injects the appropriate MEMORY_GUIDE based on active tools
        - If memory_recall_mode == "auto": perform threshold-gated auto recall
        - In hybrid mode: store query for on_output use
        """
        has_save = self._write_mode != "off"
        has_recall = self._read_mode != "off"

        if not has_save and not has_recall:
            return context

        # Store query for hybrid on_output
        if self._write_mode == "hybrid":
            self._pending_query = query

        # Choose the appropriate guide
        if has_save and has_recall:
            guide = MEMORY_GUIDE
        elif has_save:
            guide = MEMORY_GUIDE_SAVE_ONLY
        else:
            guide = MEMORY_GUIDE_RECALL_ONLY

        # Auto-recall injection
        auto_block = ""
        if self._read_mode == "auto" and self._available:
            auto_block = self._do_auto_recall(query)

        # Assemble context
        parts = []
        if context:
            parts.append(context)
        parts.append(guide)
        if auto_block:
            parts.append(auto_block)

        return "\n\n".join(parts)

    def on_output(self, response: str) -> str:
        """
        In hybrid mode, auto-extract facts from the conversation turn.

        Uses compile_hybrid() for a single LLM call combining should_store decision
        and fact extraction. Does not modify the response.
        """
        if self._write_mode != "hybrid":
            return response

        query = self._pending_query
        self._pending_query = None

        if not query or not response:
            return response

        if not self._available or self.store is None or self.compiler is None:
            return response

        try:
            result = self.compiler.compile_hybrid(query, response)
        except Exception as e:
            logger.warning("[Memory] Hybrid compilation failed: %s", e)
            return response

        if not result.should_store:
            return response

        # Process extracted facts
        if result.facts:
            self._process_facts(result.facts)
        elif result.summary:
            # No structured facts but has summary: save as episode fact
            fallback_mode = getattr(self.agent.config, "memory_fact_fallback", "text")
            if fallback_mode == "text":
                try:
                    import uuid
                    from llamagent.modules.memory.fact import MemoryFact
                    episode = MemoryFact(
                        fact_id=uuid.uuid4().hex,
                        kind="episode",
                        subject="conversation",
                        attribute="hybrid_summary",
                        value=result.summary,
                        source_text=result.summary,
                    )
                    self.store.save_fact(episode)
                except Exception as e:
                    logger.warning("[Memory] Hybrid fallback save failed: %s", e)

        return response

    # ============================================================
    # Internal helpers
    # ============================================================

    def _do_auto_recall(self, query: str) -> str:
        """
        Perform auto-recall for the current query.

        Skips short greetings. Searches memory, applies threshold gate,
        and returns a formatted block for context injection.
        """
        if not self.store or not self.store.available:
            return ""

        # Skip short greetings
        if _is_short_greeting(query):
            return ""

        max_inject = getattr(self.agent.config, "memory_auto_recall_max_inject", 3)
        threshold = getattr(self.agent.config, "memory_auto_recall_threshold", 0.35)
        top_k = getattr(self.agent.config, "memory_recall_top_k", 5)

        try:
            results = self.store.search_facts(query, top_k=top_k, status_filter="active")
        except Exception as e:
            logger.warning("[Memory] Auto-recall search failed: %s", e)
            return ""

        if not results:
            return ""

        # Apply scoring and threshold gate
        scored = self._apply_scoring(results)

        # Filter by threshold
        above_threshold = [
            item for item in scored
            if item.get("final_score", item.get("score", 0)) >= threshold
        ]

        if not above_threshold:
            return ""

        # Limit to max_inject
        injected = above_threshold[:max_inject]

        # Update access timestamps (best-effort)
        for item in injected:
            fact_id = item.get("metadata", {}).get("fact_id", item.get("id", ""))
            if fact_id:
                try:
                    self.store.update_fact_accessed(fact_id)
                except Exception:
                    pass

        # Format the auto-recall block
        memory_lines = []
        for item in injected:
            meta = item.get("metadata", {})
            kind = meta.get("kind", "")
            value = meta.get("value", item.get("text", ""))
            kind_tag = f"[{kind}] " if kind else ""
            memory_lines.append(f"- {kind_tag}{value}")

        memories_text = "\n".join(memory_lines)
        return _AUTO_RECALL_BLOCK.format(memories=memories_text)

    def _apply_scoring(self, results: list[dict]) -> list[dict]:
        """
        Apply scoring bonuses (recency, strength, kind priority) to search results.

        Returns results sorted by final_score descending.
        """
        now_ts = time.time()
        scored = []

        for item in results:
            base_score = item.get("score", 0.0)
            meta = item.get("metadata", {})

            # Recency bonus: decays over time (max +0.05 for items from last hour)
            recency_bonus = 0.0
            updated_at = meta.get("updated_at", "")
            if updated_at:
                try:
                    updated_ts = datetime.fromisoformat(updated_at).timestamp()
                    age_hours = max(0, (now_ts - updated_ts) / 3600)
                    # Exponential decay: 0.05 * e^(-age/168) (168h = 1 week)
                    recency_bonus = 0.05 * math.exp(-age_hours / 168)
                except (ValueError, TypeError, OSError):
                    pass

            # Strength bonus
            strength = float(meta.get("strength", 1.0))
            strength_bonus = (strength - 1.0) * 0.02  # Slight boost for reinforced facts

            # Kind priority bonus
            kind = meta.get("kind", "episode")
            kind_bonus = _KIND_PRIORITY.get(kind, 0.0)

            final_score = base_score + recency_bonus + strength_bonus + kind_bonus

            enriched = item.copy()
            enriched["final_score"] = round(final_score, 4)
            scored.append(enriched)

        # Sort by final_score descending
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        return scored

    def _process_facts(self, facts: list[MemoryFact]) -> str:
        """
        Merge and save a list of extracted facts.

        Returns a human-readable summary of what was saved.
        """
        saved = []
        skipped = 0
        updated = 0

        for fact in facts:
            if self.merger is not None:
                try:
                    action = self.merger.merge(fact)
                except Exception as e:
                    logger.warning("[Memory] Merge failed for fact: %s", e)
                    action = MergeAction(action="insert", fact=fact)
            else:
                action = MergeAction(action="insert", fact=fact)

            if action.action == "skip":
                # Reinforce existing fact: update access time and strength
                self.store.update_fact_accessed(action.fact.fact_id)
                skipped += 1
                continue

            if action.action == "update" and action.superseded:
                # Mark old fact as superseded
                try:
                    self.store.update_fact_status(
                        action.superseded.fact_id, "superseded"
                    )
                except Exception as e:
                    logger.warning("[Memory] Failed to supersede old fact: %s", e)
                updated += 1

            # Save the new/updated fact
            try:
                self.store.save_fact(action.fact)
                saved.append(action.fact.value)
            except Exception as e:
                logger.warning("[Memory] Failed to save fact: %s", e)

        # Build summary message
        parts = []
        if saved:
            parts.append(f"Saved {len(saved)} fact(s): {'; '.join(saved[:3])}")
            if len(saved) > 3:
                parts.append(f"(and {len(saved) - 3} more)")
        if updated:
            parts.append(f"Updated {updated} existing fact(s)")
        if skipped:
            parts.append(f"Skipped {skipped} duplicate(s)")

        if not parts:
            return "No facts were saved from the provided content."
        return ". ".join(parts) + "."

    def _save_fallback(self, content: str, category: str) -> str:
        """Save content as plain text when compiler is unavailable or fails."""
        fallback_mode = getattr(self.agent.config, "memory_fact_fallback", "text")

        if fallback_mode == "drop":
            return (
                "Could not extract structured facts from the content. "
                "Information was not saved (fact extraction required)."
            )

        # fallback_mode == "text": save as plain text
        try:
            self.store.save_memory(content, category=category)
            return f"Remembered (as text): {content}"
        except Exception as e:
            return f"Failed to save memory: {e}"

    def _handle_compile_failure(
        self, result: CompileResult, content: str, category: str
    ) -> str:
        """Handle a failed compile result according to the fallback config."""
        return self._save_fallback(content, category)

    # ============================================================
    # Programmatic interface (external use)
    # ============================================================

    @property
    def enabled(self) -> bool:
        """Check if memory has any active functionality."""
        return self._write_mode != "off" or self._read_mode != "off"

    def toggle(self, enabled: bool | None = None) -> bool:
        """
        Toggle memory on/off. Returns the state after toggling.

        When toggling off, sets both modes to "off".
        When toggling on, restores to autonomous/tool defaults.
        """
        if enabled is None:
            enabled = not self.enabled

        if enabled:
            if self._write_mode == "off":
                self._write_mode = "autonomous"
            if self._read_mode == "off":
                self._read_mode = "tool"
        else:
            self._write_mode = "off"
            self._read_mode = "off"

        return self.enabled

    def remember(self, content: str, category: str = "manual"):
        """Manually save a memory entry (bypasses compiler, saves as plain text)."""
        if self.store is None or not self.store.available:
            logger.warning("[Memory] Module has not been initialized or storage unavailable")
            return
        self.store.save_memory(content, category)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Manually search memories."""
        if self.store is None or not self.store.available:
            return []
        return self.store.search_facts(query, top_k)

    def forget_all(self):
        """Clear all memories."""
        if self.store is None:
            return
        self.store.clear()

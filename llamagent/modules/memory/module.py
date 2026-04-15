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

# FS backend guides
MEMORY_GUIDE_FS = """\
[Memory] You have long-term memory stored as files.
- Use list_memories to browse available memories (shows metadata).
- Use read_memory to read the full content of a specific memory.
- Use save_memory to save noteworthy content."""

MEMORY_GUIDE_FS_AUTO = """\
[Memory] You have long-term memory. The metadata of all active memories \
is shown below. Use read_memory to access full details of any relevant memory.
- Use save_memory to save noteworthy content."""

MEMORY_GUIDE_FS_AUTO_RECALL_ONLY = """\
[Memory] You have long-term memory. The metadata of all active memories \
is shown below. Use read_memory to access full details of any relevant memory."""

MEMORY_GUIDE_FS_RECALL_ONLY = """\
[Memory] You have long-term memory stored as files.
- Use list_memories to browse available memories (shows metadata).
- Use read_memory to read the full content of a specific memory."""

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
        self._backend: str = "rag"
        self._available: bool = False
        self._pending_query: str | None = None  # For hybrid mode on_output
        self._last_consolidation: float = 0.0  # Timestamp of last consolidation

    # ============================================================
    # Lifecycle
    # ============================================================

    def on_attach(self, agent):
        """Initialize storage backend, compiler, merger, and register tools."""
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

        # Detect backend
        self._backend = getattr(agent.config, "memory_backend", "rag")

        if self._backend == "fs":
            self._init_fs_backend(agent)
        else:
            self._init_rag_backend(agent)

        # Build compiler (needs LLM for fact extraction)
        if self._write_mode != "off":
            try:
                self.compiler = FactCompiler(self.llm)
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
            if self._backend == "fs":
                self._register_fs_recall_tools()
            else:
                self._register_recall_tool()

        # Register consolidation tool when write_mode supports it
        if self._write_mode in ("autonomous", "hybrid"):
            self._register_consolidate_tool()

    def _init_rag_backend(self, agent):
        """Initialize the RAG (ChromaDB) backend for memory storage."""
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

    def _init_fs_backend(self, agent):
        """Initialize the FS backend for memory storage."""
        import os
        from llamagent.modules.memory.fs_store import FSMemoryStore

        base_dir = getattr(agent.config, "memory_fs_dir", None)
        if not base_dir:
            base_dir = os.path.join(
                getattr(agent.config, "fs_data_dir", "data/fs"), "memory"
            )
        if agent.persona:
            base_dir = os.path.join(base_dir, agent.persona.persona_id)

        self.store = FSMemoryStore(base_dir)
        self._available = True

    def _build_pipeline(self, agent):
        """
        Build a RetrievalPipeline for memory storage via factory.

        Returns None on failure (e.g. chromadb not installed).
        Memory uses vector-only retrieval (no lexical/reranker).
        """
        try:
            from llamagent.modules.rag.factory import create_pipeline
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

    def on_input(self, user_input: str) -> str:
        """Check if memory consolidation is needed (hybrid mode auto-trigger)."""
        if self._write_mode == "hybrid" and self._should_consolidate():
            self._consolidate()

        return user_input

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

    def _register_fs_recall_tools(self):
        """Register FS backend recall tools based on read mode.

        - auto mode: only read_memory (metadata is injected via on_context)
        - tool mode: both list_memories and read_memory
        """
        if self._read_mode == "tool":
            self.agent.register_tool(
                name="list_memories",
                func=self._tool_list_memories,
                description=(
                    "List all available memories with their metadata. "
                    "Returns a summary of each memory including fact_id, kind, subject, "
                    "attribute, and value. Use this to browse what memories are available."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional keyword filter (empty string for all)",
                        },
                    },
                    "required": [],
                },
                tier="default",
                safety_level=1,
            )

        self.agent.register_tool(
            name="read_memory",
            func=self._tool_read_memory,
            description=(
                "Read the full content and source text of a specific memory by its fact_id. "
                "Use this after seeing a memory in the metadata list to get full details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": "The fact_id of the memory to read",
                    },
                },
                "required": ["fact_id"],
            },
            tier="default",
            safety_level=1,
        )

    def _register_consolidate_tool(self):
        """Register the consolidate_memory tool."""
        def consolidate_handler(**kwargs):
            return self._consolidate()

        self.agent.register_tool(
            name="consolidate_memory",
            func=consolidate_handler,
            description=(
                "Review and clean up stored memories. Removes outdated, redundant, "
                "or irrelevant memories. Call when memories feel cluttered or inaccurate."
            ),
            parameters={},
            tier="default",
            safety_level=2,
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

    def _tool_list_memories(self, query: str = "") -> str:
        """List all active memories with metadata (FS backend)."""
        if not self._available or self.store is None:
            return "Memory storage is currently unavailable."

        metadata_text = self.store.list_all_metadata()
        if not metadata_text:
            return "No memories stored yet."

        if query:
            # Simple keyword filter on the metadata text
            query_lower = query.lower()
            filtered_lines = [
                line for line in metadata_text.splitlines()
                if query_lower in line.lower()
            ]
            if not filtered_lines:
                return f"No memories matching '{query}' found."
            return "\n".join(filtered_lines)

        return metadata_text

    def _tool_read_memory(self, fact_id: str) -> str:
        """Read full content of a specific memory by fact_id (FS backend)."""
        if not self._available or self.store is None:
            return "Memory storage is currently unavailable."

        return self.store.read_fact_source(fact_id)

    # ============================================================
    # Pipeline Callbacks
    # ============================================================

    def on_context(self, query: str, context: str) -> str:
        """
        Inject memory guide and perform auto-recall if enabled.

        - Injects the appropriate MEMORY_GUIDE based on active tools and backend
        - If memory_recall_mode == "auto": perform threshold-gated auto recall (RAG)
          or inject all metadata (FS)
        - In hybrid mode: store query for on_output use
        """
        has_save = self._write_mode != "off"
        has_recall = self._read_mode != "off"

        if not has_save and not has_recall:
            return context

        # Store query for hybrid on_output
        if self._write_mode == "hybrid":
            self._pending_query = query

        # Choose the appropriate guide based on backend and modes
        if self._backend == "fs":
            if self._read_mode == "auto" and has_save:
                guide = MEMORY_GUIDE_FS_AUTO
            elif self._read_mode == "auto" and not has_save:
                guide = MEMORY_GUIDE_FS_AUTO_RECALL_ONLY
            elif has_save and has_recall:
                guide = MEMORY_GUIDE_FS
            elif has_save:
                guide = MEMORY_GUIDE_SAVE_ONLY
            else:
                guide = MEMORY_GUIDE_FS_RECALL_ONLY
        else:
            if has_save and has_recall:
                guide = MEMORY_GUIDE
            elif has_save:
                guide = MEMORY_GUIDE_SAVE_ONLY
            else:
                guide = MEMORY_GUIDE_RECALL_ONLY

        # Auto-recall injection
        auto_block = ""
        if self._read_mode == "auto" and self._available:
            if self._backend == "fs":
                raw_metadata = self.store.list_all_metadata()
                if raw_metadata:
                    auto_block = f"[Memory] Active memories:\n{raw_metadata}"
            else:
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
    # Memory consolidation
    # ============================================================

    def _consolidate(self) -> str:
        """Run memory consolidation: LLM reviews and cleans up stored memories."""
        if not self._available or self.store is None:
            return "Memory storage is currently unavailable."

        all_facts = self._load_all_active_facts()
        if not all_facts:
            return "No memories to consolidate."

        # Sort by priority: episode first, profile/preference last
        prioritized = self._prioritize_for_review(all_facts)

        total_deleted = 0
        total_updated = 0
        total_reviewed = 0
        max_deletes = max(1, int(len(all_facts) * 0.3))  # 30% cap

        for batch in self._batch(prioritized, 30):
            if total_deleted >= max_deletes:
                break  # Deletion cap reached

            total_reviewed += len(batch)
            actions = self._llm_review_batch(batch)
            for action in actions:
                if action.get("action") == "delete" and total_deleted < max_deletes:
                    self.store.update_fact_status(action["fact_id"], "archived")
                    total_deleted += 1
                elif action.get("action") == "update" and action.get("new_value"):
                    self._update_fact_value(action["fact_id"], action["new_value"])
                    total_updated += 1

        self._last_consolidation = time.time()

        summary = (
            f"Memory consolidation complete:\n"
            f"- Reviewed: {total_reviewed}\n"
            f"- Archived: {total_deleted}\n"
            f"- Updated: {total_updated}\n"
            f"- Kept: {total_reviewed - total_deleted - total_updated}"
        )
        logger.info(summary)
        return summary

    def _prioritize_for_review(self, facts: list) -> list:
        """Sort facts by cleanup priority: episodes first, profile/preference last."""
        priority_map = {
            "episode": 0,
            "project_fact": 1,
            "decision": 1,
            "preference": 2,
            "profile": 2,
            "instruction": 2,
        }
        return sorted(facts, key=lambda f: (
            priority_map.get(f.get("kind", ""), 1),
            f.get("strength", 1.0),       # Low strength first
            f.get("last_accessed_at", ""),  # Empty string = never accessed -> sorts first
        ))

    def _llm_review_batch(self, batch: list) -> list:
        """Send a batch of facts to LLM for review, return action list."""
        batch_text = "\n".join(
            f"- [{f['fact_id']}] {f['kind']}: {f['subject']}.{f['attribute']} = {f['value']} "
            f"(strength={f.get('strength', '?')}, created={f.get('created_at', '?')}, "
            f"last_accessed={f.get('last_accessed_at', 'never')})"
            for f in batch
        )

        prompt = (
            "You are reviewing stored memories for cleanup. For each memory, decide:\n"
            "- keep: still relevant and accurate\n"
            "- delete: outdated, irrelevant, or was just a casual mention\n"
            "- update: still relevant but the value needs correction (provide new_value)\n\n"
            "Context about memory kinds:\n"
            "- episode: time-sensitive events, most likely to be outdated\n"
            "- project_fact/decision: project-related, may be outdated if project is done\n"
            "- preference/profile/instruction: usually long-term valuable, be conservative\n\n"
            'Respond in JSON object: '
            '{"actions": [{"fact_id": "...", "action": "keep|delete|update", '
            '"reason": "...", "new_value": "..."}]}\n\n'
            f"Memories to review:\n{batch_text}"
        )

        result = self.llm.ask_json(prompt, temperature=0.2)

        # Defensive: handle various response formats
        if isinstance(result, dict) and "actions" in result and isinstance(result["actions"], list):
            return result["actions"]
        if isinstance(result, list):
            return result  # LLM returned actions array directly
        if isinstance(result, dict) and "raw_response" in result:
            logger.warning("Consolidation LLM returned non-JSON, skipping batch")
        return []

    def _load_all_active_facts(self) -> list:
        """Load all active facts as metadata dicts from the store."""
        return self.store.list_all_active_facts()

    def _update_fact_value(self, fact_id: str, new_value: str):
        """Update a fact's value and timestamp."""
        self.store.update_fact_value(fact_id, new_value)

    def _should_consolidate(self) -> bool:
        """Check if consolidation is needed based on time and memory count."""
        interval = getattr(self.agent.config, "memory_consolidation_interval", 24)
        if interval <= 0:
            return False  # Disabled

        now = time.time()
        if now - self._last_consolidation < interval * 3600:
            return False  # Too soon

        # Only consolidate if there are enough active memories to warrant it
        active_count = len(self._load_all_active_facts())
        min_count = getattr(self.agent.config, "memory_consolidation_min_count", 20)
        return active_count >= min_count

    @staticmethod
    def _batch(items, size):
        """Yield successive batches from items."""
        for i in range(0, len(items), size):
            yield items[i:i + size]

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

"""
MemoryModule: lets the model autonomously decide when to save and retrieve memories via tools.

Three modes:
- off:         Memory disabled, no tools registered (model cannot see save_memory / recall_memory)
- autonomous:  Fully autonomous -- model calls save_memory / recall_memory via function calling
- hybrid:      autonomous + forced write fallback -- LLM auto-summarizes after each turn, saves if valuable

The model calls via function calling:
- save_memory:   Save important information to long-term memory (ChromaDB vector store)
- recall_memory: Semantic search for relevant information from long-term memory
"""

from llamagent.core.agent import Module
from llamagent.modules.memory.backend import ChromaMemoryBackend
from llamagent.modules.memory.store import MemoryStore

# Memory usage guide injected into context, letting the model know it has this capability
MEMORY_GUIDE = """\
[Memory] You have long-term memory and can remember information across conversations.
- Use the save_memory tool to save noteworthy content
- Use the recall_memory tool to recall past information
- Do not save casual chat or meaningless content; only save truly valuable information"""

# In hybrid mode, system prompt used in on_output to determine if the conversation is worth saving
HYBRID_SUMMARY_PROMPT = """\
Determine whether the following conversation contains information worth remembering long-term (user preferences, important facts, key conclusions, work information, etc.).
Return has_value=false for casual chat or content without substance.

Please respond in JSON format:
{
  "has_value": true/false,
  "summary": "One or two sentences summarizing the key information (leave empty if has_value=false)",
  "category": "user_preference / fact / task_result / instruction (leave empty if has_value=false)"
}"""

# Valid memory modes
_VALID_MODES = ("off", "autonomous", "hybrid")


def _get_memory_mode(config) -> str:
    """
    Get the memory mode from Config.

    Prioritizes config.memory_mode (target architecture three-tier string),
    falls back to config.memory_enabled (legacy bool) for backward compatibility.
    """
    # Prefer memory_mode (if Config has been upgraded)
    mode = getattr(config, "memory_mode", None)
    if mode is not None:
        if mode in _VALID_MODES:
            return mode
        print(f"[Memory] Unknown memory_mode='{mode}', falling back to 'off'")
        return "off"

    # Fallback: infer from legacy memory_enabled (bool)
    enabled = getattr(config, "memory_enabled", False)
    return "autonomous" if enabled else "off"


class MemoryModule(Module):
    """
    Memory Module: registers memory tools, letting the model autonomously manage long-term memory.

    Behavioral differences across three modes:
    - off:         Initialize store, no tools registered, on_context/on_output do nothing
    - autonomous:  Initialize store + register tools, on_context injects memory guide
    - hybrid:      Same as autonomous + on_output forces summary and save each turn
    """

    name: str = "memory"
    description: str = "Long-term memory: can remember important information and recall it when needed"

    def __init__(self):
        self.store: MemoryStore | None = None
        self.enabled: bool = False
        self._mode: str = "off"
        self._pending_query: str | None = None  # Stores query temporarily in hybrid mode

    def on_attach(self, agent):
        """Initialize store + decide whether to register memory tools based on mode."""
        super().on_attach(agent)

        # Resolve memory mode
        self._mode = _get_memory_mode(agent.config)
        self.enabled = self._mode != "off"

        # Isolate collection by persona
        if agent.persona:
            collection = f"memory_{agent.persona.persona_id}"
        else:
            collection = "llamagent_memory"

        backend = ChromaMemoryBackend(
            persist_dir=agent.config.chroma_dir,
            collection_name=collection,
        )
        self.store = MemoryStore(backend=backend)

        # In off mode, don't register tools; model cannot see save_memory / recall_memory
        if self.enabled:
            self._register_tools()

    def _register_tools(self):
        """Register save_memory and recall_memory to the Agent (tier=default)."""
        self.agent.register_tool(
            name="save_memory",
            func=self._tool_save_memory,
            description=(
                "Save important information to long-term memory. "
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
                        "description": "Category: user_preference / fact / task_result / instruction",
                    },
                },
                "required": ["content"],
            },
            tier="default",
            safety_level=2,
        )

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
    # Tool implementation
    # ============================================================

    def _tool_save_memory(self, content: str, category: str = "general") -> str:
        """Actual execution logic for the save_memory tool."""
        if not self.enabled or self.store is None:
            return "Memory functionality is currently disabled."
        try:
            self.store.save_memory(content, category)
            return f"Remembered: {content}"
        except Exception as e:
            return f"Failed to save memory: {e}"

    def _tool_recall_memory(self, query: str) -> str:
        """Actual execution logic for the recall_memory tool."""
        if self.store is None:
            return "Memory functionality has not been initialized."
        try:
            results = self.store.recall(query, top_k=5)
        except Exception as e:
            return f"Failed to retrieve memories: {e}"

        if not results:
            return "No relevant memories found."

        lines = ["Found the following relevant memories:"]
        for r in results:
            score = r["score"]
            lines.append(f"- (relevance {score}) {r['content']}")
        return "\n".join(lines)

    # ============================================================
    # Pipeline Hooks
    # ============================================================

    def on_context(self, query: str, context: str) -> str:
        """
        Inject memory usage guide, letting the model know it has memory capabilities.

        - off mode: return the original context as-is
        - autonomous mode: inject MEMORY_GUIDE
        - hybrid mode: inject MEMORY_GUIDE + store query temporarily (for on_output use)
        """
        if not self.enabled:
            return context

        # In hybrid mode, store query temporarily for summary judgment in on_output
        if self._mode == "hybrid":
            self._pending_query = query

        return f"{context}\n\n{MEMORY_GUIDE}" if context else MEMORY_GUIDE

    def on_output(self, response: str) -> str:
        """
        In hybrid mode, force summarize and save key information from the current turn.

        Does not modify the response content; returns it as-is.
        Does nothing in autonomous / off modes.
        """
        if self._mode != "hybrid":
            return response

        # Retrieve the query stored temporarily in on_context
        query = self._pending_query
        self._pending_query = None  # Clear to avoid reuse

        if not query or not response or self.store is None:
            return response

        # Call LLM to determine if this turn's conversation is worth saving
        try:
            llm = self.agent.llm
            result = llm.ask_json(
                prompt=f"User: {query}\nAssistant: {response}",
                system=HYBRID_SUMMARY_PROMPT,
            )

            # Parse result; save if valuable
            has_value = result.get("has_value", False)
            if has_value:
                summary = result.get("summary", "")
                category = result.get("category", "conversation")
                if summary:
                    self.store.save_memory(summary, category=category)
        except Exception as e:
            # Hybrid summary failure should not affect the normal response
            print(f"[Memory] Hybrid auto-summary failed: {e}")

        return response  # Return as-is without modifying the output

    # ============================================================
    # Programmatic interface (external use)
    # ============================================================

    def toggle(self, enabled: bool | None = None) -> bool:
        """Toggle memory on/off. Returns the state after toggling."""
        if enabled is None:
            self.enabled = not self.enabled
        else:
            self.enabled = enabled
        return self.enabled

    def remember(self, content: str, category: str = "manual"):
        """Manually save a memory entry."""
        if self.store is None:
            print("[Memory] Module has not been initialized")
            return
        self.store.save_memory(content, category)

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """Manually search memories."""
        if self.store is None:
            return []
        return self.store.recall(query, top_k)

    def forget_all(self):
        """Clear all memories."""
        if self.store is None:
            return
        self.store.clear_long_term()

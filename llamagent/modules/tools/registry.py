"""
Tool Registry: manages tool registration, lookup, and schema generation.

Supports a four-tier tool system:
- default:  Core tools (meta-tools + tools registered by modules), visible to all roles
- common:   Common tools (platform built-in + admin-created), visible to all roles
- admin:    Admin-only tools, visible only to admins
- agent:    Role-specific custom tools, visible only to their creator

Core components:
- ToolInfo:        Tool information dataclass
- ToolRegistry:    Tool registry class
- global_registry: Global shared registry (module-level singleton)
- @tool decorator: Registers a regular function as an Agent tool
"""

import inspect
from dataclasses import dataclass, field
from typing import Callable, Any

from llamagent.core.agent import _infer_parameters_helper


@dataclass
class ToolInfo:
    """
    Tool information data structure.

    Difference between tier and safety_level:
    - tier:          Controls "visibility" -- whether the model can see the tool (filtered by tier + role in get_all_tool_schemas)
    - safety_level:  Core fallback only -- without SafetyModule, core blocks tools with safety_level >= 2
    """

    name: str                           # Tool name (unique identifier)
    func: Callable                      # Callable function
    description: str                    # Functional description
    parameters: dict = field(default_factory=dict)  # Parameter definition in JSON Schema format
    tier: str = "common"                # Visibility tier: "default" | "common" | "admin" | "agent"
    safety_level: int = 1               # Safety level: 1=read-only 2=has side effects 3=high risk
    creator_id: str | None = None       # Creator persona_id (only set for agent-tier tools)
    pack: str | None = None             # v1.6: pack name (None = default public surface, always visible)
    action: str | None = None           # v1.9: explicit action "read" | "write" | "execute" | None (None = infer from safety_level)
    takes_agent: bool = False           # v3.7.6: dispatcher injects the calling agent as the first positional arg before user kwargs


class ToolRegistry:
    """Tool registry: registers, manages, and looks up tools, and generates OpenAI function calling schemas."""

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    # ----------------------------------------------------------
    # Registration and removal
    # ----------------------------------------------------------

    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: dict | None = None,
        tier: str = "common",
        safety_level: int = 1,
        creator_id: str | None = None,
        pack: str | None = None,
        takes_agent: bool = False,
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name (unique identifier)
            func: Callable function
            description: Functional description, defaults to func.__doc__
            parameters: Parameter definition in JSON Schema format, defaults to inference from function signature
            tier: Visibility tier 'default' | 'common' | 'admin' | 'agent'
            safety_level: Safety level 1=read-only 2=has side effects 3=high risk
            creator_id: Creator persona_id (used for agent-tier tools)
            takes_agent: v3.7.6 — when True, the dispatcher injects the
                calling agent as the first positional arg before any
                user-supplied kwargs. Used by tools that need per-agent
                state (e.g. ``web_search`` reads the calling agent's
                ``_tool_state["web_search_backend"]``).
        """
        # v3.7.8: pass through `takes_agent` so the inferred schema skips
        # the framework-injected first arg. Pre-fix the registry's own
        # _infer_parameters didn't support skip_first_arg, so a
        # @tool(takes_agent=True) function without explicit parameters=
        # would produce a schema containing the agent param.
        self._tools[name] = ToolInfo(
            name=name,
            func=func,
            description=description or func.__doc__ or "No description",
            parameters=parameters or _infer_parameters_helper(
                func, skip_first_arg=takes_agent,
            ),
            tier=tier,
            safety_level=safety_level,
            creator_id=creator_id,
            pack=pack,
            takes_agent=takes_agent,
        )

    def remove(self, name: str) -> bool:
        """Remove a registered tool. Returns True on success, False if not found."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ----------------------------------------------------------
    # Query
    # ----------------------------------------------------------

    def get(self, name: str) -> ToolInfo | None:
        """Look up a tool by name. Returns None if not found."""
        return self._tools.get(name)

    def get_by_tier(self, *tiers: str) -> dict[str, ToolInfo]:
        """Filter tools by tier."""
        return {k: v for k, v in self._tools.items() if v.tier in tiers}

    def list_tools(self) -> list[str]:
        """Return a list of all registered tool names."""
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ----------------------------------------------------------
    # Schema / description generation
    # ----------------------------------------------------------

    def get_openai_schema(
        self,
        tiers: tuple[str, ...] | None = None,
        role: str | None = None,
    ) -> list[dict]:
        """
        Generate a tools list in OpenAI function calling format.

        Args:
            tiers: Filter by visibility tier; None means no filtering
            role: Filter by role (reserved, not currently used)
        """
        tools = self._filter(tiers)
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info.description,
                    "parameters": info.parameters,
                },
            }
            for name, info in tools.items()
        ]

    def get_descriptions(self, tiers: tuple[str, ...] | None = None) -> str:
        """Return text descriptions of tools, optionally filtered by tier. Format: - tool_name: description"""
        tools = self._filter(tiers)
        if not tools:
            return "No tools available."
        return "\n".join(
            f"- {name}: {info.description}"
            for name, info in tools.items()
        )

    # ----------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------

    def _filter(self, tiers: tuple[str, ...] | None) -> dict[str, ToolInfo]:
        """Filter tools by tier."""
        if tiers is None:
            return self._tools
        return {k: v for k, v in self._tools.items() if v.tier in tiers}

    # v3.7.8: _infer_parameters extracted to module-level
    # `core.agent._infer_parameters_helper` (shared with LlamAgent.register_tool).
    # Pre-fix this was a separate impl that did NOT support skip_first_arg
    # for takes_agent=True tools, so a @tool(takes_agent=True) function
    # without explicit parameters= would produce a schema containing
    # the framework-injected agent param. Behavior change: the previous
    # registry version added `"description": f"Parameter {pname}"` to
    # every property and always emitted `"required": []` even when empty;
    # the shared helper drops both (cleaner schema; LLM doesn't depend on
    # the placeholder description).


# ============================================================
# Global registry + @tool decorator
# ============================================================

global_registry = ToolRegistry()


def tool(
    name: str = "",
    description: str = "",
    parameters: dict | None = None,
    tier: str = "common",
    safety_level: int = 1,
    pack: str | None = None,
    takes_agent: bool = False,
):
    """
    @tool decorator: registers a function into the global tool registry.

    Usage:
        @tool(name="web_search", description="Search the web", safety_level=1)
        def web_search(query: str) -> str: ...

        @tool(tier="default", safety_level=2)
        def save_memory(...): ...

    v3.7.6: pass ``takes_agent=True`` to have the dispatcher inject the
    calling agent as the first positional arg (used by tools that read
    per-agent state from ``agent._tool_state``).
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "No description"
        global_registry.register(
            tool_name, func, tool_desc, parameters,
            tier=tier, safety_level=safety_level, pack=pack,
            takes_agent=takes_agent,
        )
        func._tool_name = tool_name
        return func
    return decorator

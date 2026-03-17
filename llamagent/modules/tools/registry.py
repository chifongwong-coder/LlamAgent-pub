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


@dataclass
class ToolInfo:
    """
    Tool information data structure.

    Difference between tier and safety_level:
    - tier:          Controls "visibility" -- whether the model can see the tool (filtered by tier + role when building prompts and schemas)
    - safety_level:  Controls "executability" -- whether the call is allowed (compares safety_level against permission_level)
    """

    name: str                           # Tool name (unique identifier)
    func: Callable                      # Callable function
    description: str                    # Functional description
    parameters: dict = field(default_factory=dict)  # Parameter definition in JSON Schema format
    tier: str = "common"                # Visibility tier: "default" | "common" | "admin" | "agent"
    safety_level: int = 1               # Safety level: 1=read-only 2=has side effects 3=high risk
    creator_id: str | None = None       # Creator persona_id (only set for agent-tier tools)


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
        """
        self._tools[name] = ToolInfo(
            name=name,
            func=func,
            description=description or func.__doc__ or "No description",
            parameters=parameters or self._infer_parameters(func),
            tier=tier,
            safety_level=safety_level,
            creator_id=creator_id,
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

    @staticmethod
    def _infer_parameters(func: Callable) -> dict:
        """Automatically infer JSON Schema parameter descriptions from function signature."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }

        sig = inspect.signature(func)
        properties = {}
        required = []

        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            json_type = (
                type_map.get(param.annotation, "string")
                if param.annotation != inspect.Parameter.empty
                else "string"
            )
            properties[pname] = {"type": json_type, "description": f"Parameter {pname}"}
            if param.default is inspect.Parameter.empty:
                required.append(pname)

        return {"type": "object", "properties": properties, "required": required}


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
):
    """
    @tool decorator: registers a function into the global tool registry.

    Usage:
        @tool(name="web_search", description="Search the web", safety_level=1)
        def web_search(query: str) -> str: ...

        @tool(tier="default", safety_level=2)
        def save_memory(...): ...
    """
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or "No description"
        global_registry.register(
            tool_name, func, tool_desc, parameters,
            tier=tier, safety_level=safety_level,
        )
        func._tool_name = tool_name
        return func
    return decorator

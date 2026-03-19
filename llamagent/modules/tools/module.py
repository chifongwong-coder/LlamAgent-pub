"""
ToolsModule: four-tier tool system + role-based permission management.

Responsibilities (tool calling logic has been moved to the core ReAct loop):
- Tool registration management: maintains global_registry and agent_registry
- Meta-tools: create_tool / list_my_tools / delete_tool / query_toolbox
- Admin tools: create_common_tool / list_all_agent_tools / promote_tool
- Tool persistence: JSON persistence for role custom tools and admin common tools

Registry separation:
- global_registry:  Globally shared, stores platform built-in tools (common tier) + admin-created common tools
- agent_registry:   Per-instance, stores meta-tools (default) + admin tools (admin) + role custom tools (agent)

Permission model:
- Super admin (human): Writes Python code to create default tools
- Admin llama (admin): Creates common tools + views all llama tools + promotes tools
- Regular llama (user): Creates custom tools + uses common tools
"""

import re

from llamagent.core.agent import Module
from llamagent.modules.tools.registry import ToolRegistry, global_registry
from llamagent.modules.tools.agent_tools import AgentToolManager


# Storage ID for admin-created common tools
COMMON_STORE_ID = "__common__"


# ----------------------------------------------------------
# Path extractors for builtin tools (used by zone system)
# ----------------------------------------------------------

def _extract_paths_from_command(args):
    """Extract absolute paths from a shell command string."""
    command = args.get("command", "")
    # Match absolute paths like /foo/bar or ~/foo
    paths = re.findall(r'(?:^|\s)([/~][^\s;|&><]+)', command)
    return [p.strip() for p in paths if p.strip()]


_BUILTIN_PATH_EXTRACTORS = {
    "read_file": lambda args: [args["filename"]] if "filename" in args else [],
    "write_file": lambda args: [args["filename"]] if "filename" in args else [],
    "execute_command": _extract_paths_from_command,
}


class ToolsModule(Module):
    """Tools module: four-tier tool system + role-based permission management."""

    name = "tools"
    description = "Tool system: core tools + common toolbox + custom tool creation"

    def __init__(self):
        self.common_registry: ToolRegistry | None = None   # Globally shared built-in common tools
        self.agent_registry: ToolRegistry = ToolRegistry()  # Per-instance: meta-tools + custom tools
        self.agent_store: AgentToolManager | None = None
        self._is_admin: bool = False

    def on_attach(self, agent):
        """Initialization logic when module is attached to an Agent."""
        super().on_attach(agent)
        self._is_admin = bool(agent.persona and agent.persona.is_admin)

        # --- 1. Load built-in tools (globally shared) ---
        import llamagent.modules.tools.builtin as builtin
        self.common_registry = global_registry

        # Bind runtime dependencies
        builtin.web_search._llm = agent.llm
        builtin.write_file._output_dir = agent.config.output_dir
        builtin.read_file._output_dir = agent.config.output_dir
        builtin.execute_command._agent = agent

        # --- 2. Load admin-created common tools (from __common__.json into common_registry) ---
        try:
            common_store = AgentToolManager(
                storage_dir=agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )
            for tool_info in common_store.list_tools():
                func = common_store.get_function(tool_info["name"])
                if func:
                    self.common_registry.register(
                        name=tool_info["name"], func=func,
                        description=tool_info["description"],
                        tier="common", safety_level=1,
                    )
        except Exception as e:
            print(f"[Tools] Failed to load common tools: {e}")

        # --- 3. Load role custom tools (per-instance) ---
        persona_id = agent.persona.persona_id if agent.persona else "default"
        try:
            self.agent_store = AgentToolManager(
                storage_dir=agent.config.agent_tools_dir,
                persona_id=persona_id,
            )
            for tool_info in self.agent_store.list_tools():
                func = self.agent_store.get_function(tool_info["name"])
                if func:
                    self.agent_registry.register(
                        name=tool_info["name"], func=func,
                        description=tool_info["description"],
                        tier="agent", safety_level=1,
                        creator_id=persona_id,
                    )
        except Exception as e:
            print(f"[Tools] Failed to load role custom tools: {e}")

        # --- 4. Register meta-tools (per-instance, by role) ---
        self._register_meta_tools()

        # --- 5. Bridge all tools to agent._tools (core visibility) ---
        self._bridge_to_core()


    # ============================================================
    # Bridge internal registries → agent._tools (core visibility)
    # ============================================================

    def _bridge_to_core(self):
        """Sync tools from internal registries to agent._tools so the LLM can see and call them."""
        for _name, info in self.common_registry._tools.items():
            if _name in self.agent._tools:
                continue  # Don't overwrite tools registered by other modules
            self.agent.register_tool(
                name=info.name, func=info.func, description=info.description,
                parameters=info.parameters, tier=info.tier,
                safety_level=info.safety_level,
                path_extractor=_BUILTIN_PATH_EXTRACTORS.get(info.name),
            )
        for _name, info in self.agent_registry._tools.items():
            if _name in self.agent._tools:
                continue
            self.agent.register_tool(
                name=info.name, func=info.func, description=info.description,
                parameters=info.parameters, tier=info.tier,
                safety_level=info.safety_level,
                creator_id=info.creator_id,
            )

    # ============================================================
    # Meta-tool registration (all go to agent_registry to avoid global overwrite)
    # ============================================================

    def _register_meta_tools(self):
        """Register meta-tools: shared across all roles + admin-only."""

        # --- Shared across all roles (default tier) ---
        self.agent_registry.register(
            name="create_tool",
            func=self._tool_create,
            description="Create a custom tool. Provide the tool name, description, and Python function code.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Tool function name (in English)"},
                    "description": {"type": "string", "description": "Functional description"},
                    "code": {"type": "string", "description": "Python function code; the function name must match the name parameter"},
                },
                "required": ["name", "description", "code"],
            },
            tier="default",
            safety_level=2,
        )

        self.agent_registry.register(
            name="list_my_tools",
            func=self._tool_list_my_tools,
            description="View the list of custom tools you have created.",
            parameters={"type": "object", "properties": {}},
            tier="default",
            safety_level=1,
        )

        self.agent_registry.register(
            name="delete_tool",
            func=self._tool_delete,
            description="Delete one of your custom tools.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Tool name"}},
                "required": ["name"],
            },
            tier="default",
            safety_level=2,
        )

        self.agent_registry.register(
            name="query_toolbox",
            func=self._tool_query_toolbox,
            description="Query the list of common tools in the toolbox.",
            parameters={"type": "object", "properties": {}},
            tier="default",
            safety_level=1,
        )

        # --- Admin-only (admin tier) ---
        if self._is_admin:
            self.agent_registry.register(
                name="create_common_tool",
                func=self._tool_create_common,
                description="[Admin] Create a common tool available to all llamas.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Tool function name"},
                        "description": {"type": "string", "description": "Functional description"},
                        "code": {"type": "string", "description": "Python function code"},
                    },
                    "required": ["name", "description", "code"],
                },
                tier="admin",
                safety_level=2,
            )

            self.agent_registry.register(
                name="list_all_agent_tools",
                func=self._tool_list_all_agent_tools,
                description="[Admin] View custom tools created by all llamas.",
                parameters={"type": "object", "properties": {}},
                tier="admin",
                safety_level=1,
            )

            self.agent_registry.register(
                name="promote_tool",
                func=self._tool_promote,
                description="[Admin] Promote a llama's custom tool to a common tool.",
                parameters={
                    "type": "object",
                    "properties": {
                        "persona_id": {"type": "string", "description": "Llama ID"},
                        "tool_name": {"type": "string", "description": "Tool name"},
                    },
                    "required": ["persona_id", "tool_name"],
                },
                tier="admin",
                safety_level=2,
            )

    # ============================================================
    # Tool implementations: shared across all roles
    # ============================================================

    def _tool_create(self, name: str, description: str, code: str) -> str:
        """Create a role custom tool."""
        if self.agent_store is None:
            return "Tool storage not initialized, cannot create tool."

        try:
            # If the safety module is available, scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(code)

            self.agent_store.create(name, description, code)
            func = self.agent_store.get_function(name)
            if func:
                persona_id = self.agent.persona.persona_id if self.agent.persona else "default"
                self.agent_registry.register(
                    name=name, func=func, description=description,
                    tier="agent", safety_level=safety_level,
                    creator_id=persona_id,
                )
                # Bridge to core so LLM can see and call the new tool
                self.agent.register_tool(
                    name=name, func=func, description=description,
                    tier="agent", safety_level=safety_level,
                    creator_id=persona_id,
                )
            return f"Tool '{name}' created successfully! (safety level: {safety_level})"
        except ValueError as e:
            return f"Creation failed: {e}"
        except Exception as e:
            return f"Unexpected error while creating tool: {e}"

    def _tool_list_my_tools(self) -> str:
        """View the list of custom tools created by the current role."""
        if self.agent_store is None:
            return "Tool storage not initialized."

        tools = self.agent_store.list_tools()
        if not tools:
            return "You haven't created any custom tools yet."
        lines = ["Your custom tools:"]
        for t in tools:
            lines.append(f"- {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _tool_delete(self, name: str) -> str:
        """Delete a custom tool of the current role."""
        if self.agent_store is None:
            return "Tool storage not initialized."

        if self.agent_store.delete(name):
            self.agent_registry.remove(name)
            self.agent.remove_tool(name)  # Remove from core
            return f"Tool '{name}' has been deleted."
        return f"No custom tool named '{name}' found."

    def _tool_query_toolbox(self) -> str:
        """View the common toolbox (common-tier tools only)."""
        if self.common_registry is None:
            return "Tool registry not initialized."

        common = self.common_registry.get_by_tier("common")
        if not common:
            return "No common tools in the toolbox yet."
        lines = ["Common tools in the toolbox:"]
        for name, info in common.items():
            lines.append(f"- {name}: {info.description}")
        return "\n".join(lines)

    # ============================================================
    # Tool implementations: admin-only
    # ============================================================

    def _tool_create_common(self, name: str, description: str, code: str) -> str:
        """Create a common tool, registered to global_registry, available to all roles."""
        try:
            common_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )

            # Scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(code)

            common_store.create(name, description, code)
            func = common_store.get_function(name)
            if func:
                self.common_registry.register(
                    name=name, func=func, description=description,
                    tier="common", safety_level=safety_level,
                )
                # Bridge to core
                self.agent.register_tool(
                    name=name, func=func, description=description,
                    tier="common", safety_level=safety_level,
                )
            return f"Common tool '{name}' created successfully! Available to all llamas. (safety level: {safety_level})"
        except ValueError as e:
            return f"Creation failed: {e}"
        except Exception as e:
            return f"Unexpected error while creating common tool: {e}"

    def _tool_list_all_agent_tools(self) -> str:
        """View custom tools created by all roles, grouped by persona_id."""
        try:
            all_tools = AgentToolManager.scan_all(self.agent.config.agent_tools_dir)
        except Exception as e:
            return f"Failed to scan tools directory: {e}"

        if not all_tools:
            return "No llamas have created any custom tools yet."
        lines = ["Custom tools from all llamas:"]
        for pid, tools in all_tools.items():
            lines.append(f"\n[{pid}]")
            for t in tools:
                lines.append(f"  - {t['name']}: {t['description']}")
        return "\n".join(lines)

    def _tool_promote(self, persona_id: str, tool_name: str) -> str:
        """Promote a specified role's custom tool to a common tool."""
        try:
            source_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=persona_id,
            )
        except Exception as e:
            return f"Failed to load tool storage for role [{persona_id}]: {e}"

        tool_def = source_store.export(tool_name)
        if not tool_def:
            return f"Tool '{tool_name}' not found in [{persona_id}]."

        try:
            common_store = AgentToolManager(
                storage_dir=self.agent.config.agent_tools_dir,
                persona_id=COMMON_STORE_ID,
            )

            # Scan the code to determine safety_level
            safety_level = 1
            safety_mod = self.agent.get_module("safety")
            if safety_mod and hasattr(safety_mod, "guard"):
                safety_level = safety_mod.guard.scan_code(tool_def.get("code", ""))

            common_store.create(
                name=tool_def["name"],
                description=tool_def["description"],
                code=tool_def["code"],
                parameters=tool_def.get("parameters"),
            )
            func = common_store.get_function(tool_def["name"])
            if func:
                self.common_registry.register(
                    name=tool_def["name"], func=func,
                    description=tool_def["description"],
                    tier="common", safety_level=safety_level,
                )
                # Bridge to core
                self.agent.register_tool(
                    name=tool_def["name"], func=func,
                    description=tool_def["description"],
                    tier="common", safety_level=safety_level,
                )
            return f"Tool '{tool_name}' has been promoted from [{persona_id}] to a common tool! (safety level: {safety_level})"
        except ValueError as e:
            return f"Promotion failed: {e}"
        except Exception as e:
            return f"Unexpected error while promoting tool: {e}"

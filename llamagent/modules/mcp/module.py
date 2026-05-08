"""
MCPModule: MCP external integration module.

Capabilities:
- Connect to MCP Servers, auto-discover and register external tools
- Support both stdio and SSE transport modes
- Transparently bridge MCP tools as locally available Agent tools

Tool registration method:
- Registered via agent.register_tool() (tier=default), independent of the tools module
- Bridge includes complete parameters schema to ensure correct function calling parameter passing

Configuration:
  Set the MCP_SERVERS environment variable to a JSON string, e.g.:
  MCP_SERVERS='{"weather": {"transport": "stdio", "command": "python", "args": ["-m", "llamagent.modules.mcp.server_example"]}}'
"""

import os
import json
import asyncio
import logging

from llamagent.core.agent import Module

logger = logging.getLogger(__name__)

# MCP package is an optional dependency, only imported when actually used
_MCP_INSTALL_HINT = "[MCP] mcp package not installed, please run: pip install mcp"


class MCPModule(Module):
    """MCP external integration module: connect to MCP Servers, bridge remote tools as locally callable functions."""

    name = "mcp"
    description = "MCP external integration: connect to external system tools and services"

    # v3.7.8: declare shareable so child agents that explicitly request it
    # (`AgentExecutionPolicy.share_parent_modules=["mcp"]`) reuse the
    # parent's MCPClient. Default is still NOT to share — the connection
    # lifecycle is parent-bound (parent shutdown disconnects child too),
    # so opt-in carries a documented caveat.
    shareable = True

    def __init__(self):
        self.client = None
        self._connected: bool = False
        # v3.7.8: track bridged tool names so child agent factory can
        # strip parent-bound closures. Static ClassVar isn't usable here
        # because tool names depend on connected servers.
        self._bridged_tool_names: set[str] = set()

    def on_attach(self, agent):
        """
        Initialize MCP client and bridge tools.

        Flow:
        1. Save agent reference (used by _bridge_tools)
        2. Read MCP server configuration from environment variables
        3. Create MCPClient and attempt connection
        4. Bridge tools to tool registry after successful connection
        """
        super().on_attach(agent)

        # Read MCP server configuration from environment variables
        mcp_config = os.getenv("MCP_SERVERS")
        if not mcp_config:
            return

        try:
            server_configs = json.loads(mcp_config)
        except json.JSONDecodeError:
            logger.warning("[MCP] MCP_SERVERS environment variable has invalid JSON format, please check configuration")
            return

        self._init_client(server_configs)

    def _init_client(self, server_configs: dict) -> None:
        """Initialize MCP client and connect to all configured servers.

        v3.8.1 R7-#2: connect_all runs on the client's persistent
        background event loop (separate thread). This works identically
        whether the caller is in a sync context or already inside an
        outer asyncio loop (FastAPI / Gradio / Jupyter) — pre-fix the
        per-call ``asyncio.run`` left sessions referencing destroyed loops.
        """
        try:
            from llamagent.modules.mcp.client import MCPClient, MCP_AVAILABLE

            if not MCP_AVAILABLE:
                logger.info(_MCP_INSTALL_HINT)
                return

            self.client = MCPClient(server_configs)

            # Start the persistent loop BEFORE submitting any coroutine.
            # Sessions opened via connect_all() will live on this loop;
            # subsequent call_tool / disconnect must run on the same loop.
            self.client._loop.start()
            try:
                results = self.client._loop.submit(
                    self.client.connect_all(), timeout=30
                )
            except Exception as e:
                logger.warning("[MCP] connection failed: %s", e)
                return

            self._connected = any(results.values())
            if self._connected:
                self._bridge_tools()
            else:
                logger.warning("[MCP] All server connections failed, please check configuration")

        except ImportError:
            logger.info(_MCP_INSTALL_HINT)
        except Exception as e:
            logger.warning("[MCP] Initialization failed: %s", e)

    def _bridge_tools(self) -> None:
        """
        Bridge MCP tools to the tool registry.

        Independent of the tools module, registers directly via agent_registry.
        Includes parameters schema during bridging to ensure correct function calling parameter passing.
        """
        if not self.client:
            return

        try:
            from llamagent.modules.mcp.client import MCPToolBridge
        except ImportError:
            logger.info(_MCP_INSTALL_HINT)
            return

        # Get bridge functions and parameter schemas
        bridge = MCPToolBridge(self.client)
        bridged = bridge.get_bridged_tools()
        schemas = self.client.get_tools_as_functions()

        # Build name -> parameters mapping
        param_map = {
            s["function"]["name"]: s["function"]["parameters"]
            for s in schemas
        }

        # Register bridged tools to Agent one by one
        for name, func in bridged.items():
            self.agent.register_tool(
                name=name,
                func=func,
                description=func.__doc__ or f"MCP tool: {name}",
                parameters=param_map.get(name, {}),
                tier="default",
            )
            # v3.7.8: track for child factory closure-strip
            self._bridged_tool_names.add(name)

        tool_count = len(bridged)
        if tool_count > 0:
            logger.info("[MCP] Bridged %d tools to registry", tool_count)

    @property
    def _service_bound_tool_names(self) -> set[str]:
        """v3.7.8: MCP tool names depend on connected servers — return the
        live set tracked by ``_bridge_tools``. Used by child agent factory
        to strip parent-bound closures from ``child._tools``."""
        return set(self._bridged_tool_names)

    def inherit_storage_from(self, parent_mod: "MCPModule") -> None:
        """v3.7.8: child agent factory calls this when
        ``share_parent_modules=["mcp"]`` is explicitly opted into.

        Reuses the parent's MCPClient + connection state. **Caveat**: the
        client holds stdio subprocess handles owned by the parent agent;
        when the parent shuts down, the connection drops for the child too.
        Callers must ensure the parent outlives the child or accept the
        coupling.
        """
        self.client = parent_mod.client
        self._connected = parent_mod._connected
        self._bridged_tool_names = set(parent_mod._bridged_tool_names)

    # ============================================================
    # Lifecycle
    # ============================================================

    def on_shutdown(self) -> None:
        """Disconnect all MCP server connections and release resources.

        v3.8.1 R7-#2 canonical shutdown ordering:
            1. submit ``disconnect_all`` onto the persistent loop (where
               sessions were opened) and wait for the future
            2. ``call_soon_threadsafe(loop.stop)``
            3. ``thread.join(timeout=5)``

        Doing disconnect on a fresh short-lived loop is REJECTED — would
        re-introduce the very "different event loop" RuntimeError this
        fix closes (see ``_loop.py`` module docstring).
        """
        if self.client is None:
            return

        try:
            # Step 1: disconnect on persistent loop (sessions live there)
            try:
                self.client._loop.submit(
                    self.client.disconnect_all(), timeout=5
                )
            except Exception as e:
                logger.warning(
                    "[MCP] disconnect_all on persistent loop failed: %s", e
                )
            # Steps 2-3: stop loop + join thread
            self.client._loop.stop(join_timeout=5.0)

            self._connected = False
            logger.info("[MCP] All connections disconnected")

        except Exception as e:
            logger.warning("[MCP] Error during shutdown: %s", e)

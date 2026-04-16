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

from llamagent.core.agent import Module

# MCP package is an optional dependency, only imported when actually used
_MCP_INSTALL_HINT = "[MCP] mcp package not installed, please run: pip install mcp"


class MCPModule(Module):
    """MCP external integration module: connect to MCP Servers, bridge remote tools as locally callable functions."""

    name = "mcp"
    description = "MCP external integration: connect to external system tools and services"

    def __init__(self):
        self.client = None
        self._connected: bool = False

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
            print("[MCP] MCP_SERVERS environment variable has invalid JSON format, please check configuration")
            return

        self._init_client(server_configs)

    def _init_client(self, server_configs: dict) -> None:
        """Initialize MCP client and connect to all configured servers."""
        try:
            from llamagent.modules.mcp.client import MCPClient, MCP_AVAILABLE

            if not MCP_AVAILABLE:
                print(_MCP_INSTALL_HINT)
                return

            self.client = MCPClient(server_configs)

            # Detect if already in async environment
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Async environment: run connect_all() in a separate thread to avoid
                # event loop conflict. The 30s blocking is acceptable for one-time
                # startup during on_attach (not called in hot path).
                from concurrent.futures import ThreadPoolExecutor
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        results = pool.submit(lambda: asyncio.run(self.client.connect_all())).result(timeout=30)
                    self._connected = any(results.values())
                    if self._connected:
                        self._bridge_tools()
                    else:
                        print("[MCP] All server connections failed, please check configuration")
                except Exception as e:
                    print(f"[MCP] Async environment connection failed: {e}")
            else:
                # Synchronous environment, connect immediately
                results = asyncio.run(self.client.connect_all())
                self._connected = any(results.values())
                if self._connected:
                    self._bridge_tools()
                else:
                    print("[MCP] All server connections failed, please check configuration")

        except ImportError:
            print(_MCP_INSTALL_HINT)
        except Exception as e:
            print(f"[MCP] Initialization failed: {e}")

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
            print(_MCP_INSTALL_HINT)
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

        tool_count = len(bridged)
        if tool_count > 0:
            print(f"[MCP] Bridged {tool_count} tools to registry")

    # ============================================================
    # Lifecycle
    # ============================================================

    def on_shutdown(self) -> None:
        """Disconnect all MCP Server connections and release resources."""
        if self.client is None:
            return

        try:
            # Detect event loop environment
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # In async environment, disconnect via thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.client.disconnect_all(),
                    )
                    future.result(timeout=10)
            else:
                asyncio.run(self.client.disconnect_all())

            self._connected = False
            print("[MCP] All connections disconnected")

        except Exception as e:
            print(f"[MCP] Error during disconnection: {e}")

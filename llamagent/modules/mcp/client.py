"""
MCP client: manages connections to MCP Servers and tool bridging.

Core capabilities:
1. Connect to MCP Servers (supports both stdio and SSE transport modes)
2. Discover tools provided by the server
3. Call remote tools and retrieve results (with retry mechanism)
4. Manage connection lifecycle for multiple MCP Servers
5. Bridge MCP tools as locally callable synchronous functions

Dependency: The mcp package is an optional dependency; the module degrades gracefully when not installed.
"""

import json
import asyncio
from typing import Any, Callable

# MCP package is an optional dependency; graceful degradation when not installed
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None


class MCPClient:
    """
    MCP client manager: manages connections to multiple MCP Servers.

    Supports both stdio and SSE transport modes, auto-discovers tools and provides remote invocation capabilities.
    Supports async with context manager (auto connect_all / disconnect_all).

    Usage example::

        client = MCPClient(server_configs)
        await client.connect_all()
        tools = await client.list_all_tools()
        result = await client.call_tool("weather", "get_weather", {"city": "Beijing"})
        await client.disconnect_all()

    Or using context manager::

        async with MCPClient(server_configs) as client:
            result = await client.call_tool("weather", "get_weather", {"city": "Beijing"})
    """

    def __init__(
        self,
        server_configs: dict,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the MCP client.

        Args:
            server_configs: MCP server configuration dictionary, format:
                {
                    "server_name": {
                        "transport": "stdio",        # or "sse"
                        "command": "python",          # command for stdio mode
                        "args": ["server.py"],        # arguments for stdio mode
                        "url": "http://...",          # URL for sse mode
                        "headers": {},                # request headers for sse mode (optional)
                        "env": {},                    # environment variables (optional)
                    }
                }
            timeout: Per-call timeout in seconds
            max_retries: Maximum number of retries
        """
        self.server_configs = server_configs
        self.timeout = timeout
        self.max_retries = max_retries
        # Active sessions: {server_name: ClientSession}
        self._sessions: dict[str, Any] = {}
        # Connection context managers (for cleanup)
        self._connections: dict[str, Any] = {}
        # Tool list cache: {server_name: [tool_object_list]}
        self._tools_cache: dict[str, list] = {}

    # ============================================================
    # Connection Management
    # ============================================================

    async def connect(self, server_name: str) -> bool:
        """
        Connect to the specified MCP Server.

        Args:
            server_name: Server name (corresponding key in configuration)

        Returns:
            Whether the connection was successful
        """
        if not MCP_AVAILABLE:
            print("[MCP] mcp package not installed, please run: pip install mcp")
            return False

        if server_name not in self.server_configs:
            print(f"[MCP] Error: server configuration '{server_name}' not found")
            return False

        if server_name in self._sessions:
            print(f"[MCP] Server '{server_name}' is already connected")
            return True

        config = self.server_configs[server_name]
        transport = config.get("transport", "stdio")

        try:
            if transport == "stdio":
                session = await self._connect_stdio(server_name, config)
            elif transport == "sse":
                session = await self._connect_sse(server_name, config)
            else:
                print(f"[MCP] Unsupported transport mode: {transport}")
                return False

            self._sessions[server_name] = session
            print(f"[MCP] Connected to server: {server_name}")

            # Discover tools immediately after successful connection
            tools = await self._discover_tools(server_name)
            print(f"[MCP] Discovered {len(tools)} tools: {[t.name for t in tools]}")

            return True

        except Exception as e:
            print(f"[MCP] Failed to connect to server '{server_name}': {e}")
            return False

    async def _connect_stdio(self, name: str, config: dict) -> "ClientSession":
        """
        Connect to an MCP Server via stdio (standard input/output).
        Suitable for locally running MCP Servers.
        """
        server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env"),
        )

        # Create stdio connection
        client_ctx = stdio_client(server_params)
        client = await client_ctx.__aenter__()
        self._connections[name] = client_ctx
        read, write = client

        # Initialize MCP session (protocol handshake)
        session = ClientSession(read, write)
        await asyncio.wait_for(session.initialize(), timeout=self.timeout)

        return session

    async def _connect_sse(self, name: str, config: dict) -> "ClientSession":
        """
        Connect to a remote MCP Server via SSE (Server-Sent Events).
        Suitable for remotely deployed MCP Servers.
        """
        url = config["url"]
        headers = config.get("headers", {})

        # Create SSE connection
        client_ctx = sse_client(url, headers=headers)
        client = await client_ctx.__aenter__()
        self._connections[name] = client_ctx
        read, write = client

        # Initialize session
        session = ClientSession(read, write)
        await asyncio.wait_for(session.initialize(), timeout=self.timeout)

        return session

    async def connect_all(self) -> dict[str, bool]:
        """
        Connect to all configured MCP Servers.

        Returns:
            {server_name: whether_connection_successful}
        """
        results = {}
        for name in self.server_configs:
            results[name] = await self.connect(name)
        return results

    async def disconnect(self, server_name: str) -> None:
        """Disconnect from the specified MCP Server."""
        # Close session first
        if server_name in self._sessions:
            try:
                session = self._sessions.pop(server_name)
                if hasattr(session, "__aexit__"):
                    await session.__aexit__(None, None, None)
            except Exception as e:
                print(f"[MCP] Error disconnecting session '{server_name}': {e}")

        # Then close underlying connection
        if server_name in self._connections:
            try:
                ctx = self._connections.pop(server_name)
                await ctx.__aexit__(None, None, None)
            except Exception as e:
                print(f"[MCP] Error cleaning up connection '{server_name}': {e}")

        self._tools_cache.pop(server_name, None)
        print(f"[MCP] Disconnected: {server_name}")

    async def disconnect_all(self) -> None:
        """Disconnect all MCP Server connections."""
        server_names = list(self._sessions.keys())
        for name in server_names:
            await self.disconnect(name)

    # ============================================================
    # Tool Discovery and Invocation
    # ============================================================

    async def _discover_tools(self, server_name: str) -> list:
        """Discover the tools provided by the specified server."""
        session = self._sessions.get(server_name)
        if not session:
            return []

        response = await session.list_tools()
        tools = response.tools
        self._tools_cache[server_name] = tools
        return tools

    async def list_all_tools(self) -> dict[str, list]:
        """
        List tools from all connected servers.

        Returns:
            {server_name: [tool_list]}
        """
        all_tools = {}
        for name in self._sessions:
            if name not in self._tools_cache:
                await self._discover_tools(name)
            all_tools[name] = self._tools_cache.get(name, [])
        return all_tools

    def get_tools_as_functions(self) -> list[dict]:
        """
        Convert all MCP tools to OpenAI function calling format.

        Tool name format: servername__toolname (double underscore separated).
        Includes complete parameters schema to ensure correct parameter passing for function calling.

        Returns:
            Tool list in OpenAI function calling format
        """
        functions = []
        for server_name, tools in self._tools_cache.items():
            for tool in tools:
                functions.append({
                    "type": "function",
                    "function": {
                        "name": f"{server_name}__{tool.name}",
                        "description": f"[{server_name}] {tool.description}",
                        "parameters": tool.inputSchema,
                    },
                })
        return functions

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict,
    ) -> str:
        """
        Call a tool on the specified server (with retry mechanism).

        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool parameters

        Returns:
            Tool execution result (string)
        """
        session = self._sessions.get(server_name)
        if not session:
            return f"Error: not connected to server '{server_name}'"

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=self.timeout,
                )

                # Extract text content
                if result.content:
                    texts = [
                        item.text
                        for item in result.content
                        if hasattr(item, "text")
                    ]
                    return "\n".join(texts)
                return "Tool executed successfully but returned no content"

            except asyncio.TimeoutError:
                print(
                    f"[MCP] Call timed out (attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt == self.max_retries - 1:
                    return f"Error: call to tool '{tool_name}' timed out"

            except Exception as e:
                print(
                    f"[MCP] Call error (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt == self.max_retries - 1:
                    return f"Error: call to tool '{tool_name}' failed - {e}"

        # Fallback after for loop ends (when max_retries=0)
        return f"Error: tool call not executed (max_retries={self.max_retries})"

    async def call_tool_by_full_name(
        self, full_name: str, arguments: dict
    ) -> str:
        """
        Call a tool by its full name (format: servername__toolname).

        Convenient for integration with OpenAI function calling,
        since get_tools_as_functions() generates tool names in this format.
        """
        if "__" not in full_name:
            return (
                f"Error: invalid tool name format '{full_name}', "
                f"expected format: servername__toolname"
            )

        server_name, tool_name = full_name.split("__", 1)
        return await self.call_tool(server_name, tool_name, arguments)

    # ============================================================
    # Context Manager Support
    # ============================================================

    async def __aenter__(self):
        """Support async with syntax, automatically connects to all servers."""
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Automatically disconnects all connections on exit."""
        await self.disconnect_all()


class MCPToolBridge:
    """
    MCP tool bridge: wraps async MCP remote tools as synchronous locally callable functions.

    Bridge functions automatically handle async/sync environment switching:
    - No event loop: directly use asyncio.run()
    - Running event loop present: execute via thread pool

    Usage example::

        bridge = MCPToolBridge(mcp_client)
        bridged = bridge.get_bridged_tools()  # {full_tool_name: sync_callable}
    """

    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client

    def get_bridged_tools(self) -> dict[str, Callable]:
        """
        Generate bridge functions for all MCP tools.

        Returns:
            {full_tool_name: sync_callable}, full tool name format is servername__toolname
        """
        bridged = {}

        for server_name, tools in self.mcp_client._tools_cache.items():
            for tool in tools:
                func = self._create_bridge_func(server_name, tool)
                bridged_name = f"{server_name}__{tool.name}"
                bridged[bridged_name] = func

        return bridged

    def _create_bridge_func(self, server_name: str, tool) -> Callable:
        """
        Create a synchronous bridge function for a single MCP tool.

        Uses closure to capture server_name and tool, ensuring each bridge function
        remembers its corresponding server and tool.
        """
        mcp_client = self.mcp_client

        def bridge_func(**kwargs) -> str:
            """Synchronous wrapper: calls async MCP tool, automatically handles event loop environment."""
            try:
                # Check if there's already a running event loop
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async environment, use thread pool to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        mcp_client.call_tool(server_name, tool.name, kwargs),
                    )
                    return future.result(timeout=mcp_client.timeout)
            else:
                # No event loop, create a new one
                return asyncio.run(
                    mcp_client.call_tool(server_name, tool.name, kwargs)
                )

        # Set function metadata (used by tool registry)
        bridge_func.__doc__ = f"[{server_name}] {tool.description}"
        bridge_func.__name__ = f"{server_name}__{tool.name}"

        return bridge_func

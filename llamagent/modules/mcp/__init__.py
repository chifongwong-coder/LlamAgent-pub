"""MCP module: connect to external systems via MCP protocol, bridge remote tools as locally callable functions."""

from llamagent.modules.mcp.module import MCPModule

__all__ = ["MCPModule"]

# MCPClient and MCPToolBridge are imported on demand (to avoid errors when mcp is not installed)
# Usage: from llamagent.modules.mcp.client import MCPClient, MCPToolBridge

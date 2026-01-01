from __future__ import annotations as _annotations

import os

from pydantic_ai.mcp import MCPServerStreamableHTTP

# Create a global MCP server instance to avoid repeated connections
_mcp_server = None


def get_mcp_server():
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServerStreamableHTTP(
            os.getenv("DDIFFUSION_MCP_ADDRESS", "http://localhost:5001/mcp"),
            headers={"Authorization": f"Bearer {os.getenv('DDIFFUSION_API_KEY')}"},
        )
    return _mcp_server

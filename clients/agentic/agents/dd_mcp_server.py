import os

from pydantic_ai.mcp import MCPServerStreamableHTTP


def get_mcp_server():
    return MCPServerStreamableHTTP(
        os.getenv("DDIFFUSION_MCP_ADDRESS", "http://localhost:5000/mcp"),
        headers={"Authorization": f"Bearer {os.getenv('DDIFFUSION_API_KEY')}"},
    )

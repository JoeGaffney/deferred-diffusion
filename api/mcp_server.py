from fastmcp import FastMCP

from main import app

# Create MCP server with authentication headers
mcp = FastMCP.from_fastapi(
    app=app,
    httpx_client_kwargs={
        "headers": {
            "Authorization": "Bearer secret-token",
        }
    },
)


@mcp.tool
def greet(name: str) -> str:
    return f"Hello, {name}!"


# # Create ASGI app from MCP server
# mcp_app = mcp.http_app(path="/mcp")

# Run the MCP server
if __name__ == "__main__":
    mcp.run()

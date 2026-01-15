from pydantic_ai import Agent

from .dd_mcp_server import get_mcp_server

chat_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You are are an expert in AI image and video generation. "
        "You help the user by generating images and videos based on their prompts."
    ),
    toolsets=[get_mcp_server()],
    retries=1,
)

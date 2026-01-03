from __future__ import annotations as _annotations

import asyncio

from pydantic_ai import Agent

from common.logger import logger
from tools.dd_mcp_server import get_mcp_server

chat_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You are are an expert in AI image and video generation. "
        "You help the user by generating images and videos based on their prompts."
    ),
    toolsets=[get_mcp_server()],
    retries=1,
)


async def main():
    result = await chat_agent.run("What image models are available?")
    logger.info("Response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())

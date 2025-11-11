from __future__ import annotations as _annotations

import asyncio

from pydantic_ai import Agent

from common.logger import logger
from common.state import Deps
from tools.dd_mcp_server import get_mcp_server

chat_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You are are an expert in AI image and video generation. "
        "You help the user by generating images and videos based on their prompts."
    ),
    deps_type=Deps,
    toolsets=[get_mcp_server()],
    retries=1,
)


async def main():
    deps = Deps()
    result = await chat_agent.run("What image models are available?", deps=deps)
    logger.info("Response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())

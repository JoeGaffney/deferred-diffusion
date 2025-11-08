from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent

from common.logger import logger
from tools.dd_mcp_server import get_mcp_server
from utils.utils import CACHE_DIR


@dataclass
class Deps:
    client: None


chat_agent = Agent(
    "openai:gpt-4.1-mini",
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    # system_prompt=(
    #     "You are are an expert in AI image and video generation "
    #     "When retrieving images or videos do this as the end of the conversation and do incremental back off max 3 times "
    # ),
    instructions=(
        "You are are an expert in AI image and video generation. "
        "You help the user by generating images and videos based on their prompts. "
        # "When retrieving 'images_get' or 'videos_get' do this as the end of the conversation and do with an incremental back off max 3 times. As sometimes they can take time to generate."
        # "You give guidance on how to improve prompts for better results. "
        # "You know about various AI image and video generation models and their capabilities. "
        # "Some are local and some are external models."
        # "You can help the user edit existing images and videos as well."
    ),
    deps_type=Deps,
    toolsets=[get_mcp_server()],
    retries=1,
)


async def main():
    deps = Deps(client=None)
    result = await chat_agent.run("What image models are available?", deps=deps)
    logger.info("Response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())

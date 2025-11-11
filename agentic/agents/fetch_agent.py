from __future__ import annotations as _annotations

import asyncio

from pydantic_ai import Agent, RunContext

from common.logger import logger
from common.state import Deps
from tools.dd_mcp_server import get_mcp_server

fetch_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You retrieve pending images and videos by using the get_pending_task_ids. "
        "You then call images_get and videos_get to check their status and retrieve them when ready."
    ),
    deps_type=Deps,
    toolsets=[get_mcp_server()],
    retries=1,
)


@fetch_agent.tool
def get_pending_task_ids(ctx: RunContext[Deps]) -> list[str]:
    return ctx.deps.get_pending_task_ids()

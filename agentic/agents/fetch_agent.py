from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ToolReturnPart

from tools.dd_mcp_server import get_mcp_server

fetch_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You retrieve pending images and videos by first calling get_pending_task_ids. "
        "Then for each ID, call images_get or videos_get to check their status. "
        "Provide details of model and prompt used when returning media."
    ),
    deps_type=list[ModelMessage],
    toolsets=[get_mcp_server()],
    retries=1,
)


@fetch_agent.tool
def get_pending_task_ids(ctx: RunContext[list[ModelMessage]]) -> list[str]:
    """
    Scans the message history for image/video generation tasks that are not yet successful or failed.
    Returns a list of task IDs.
    """
    tasks: dict[str, str] = {}

    for msg in ctx.deps:
        if not hasattr(msg, "parts"):
            continue

        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                # Extract content
                content = part.content
                if hasattr(content, "model_dump"):
                    content = content.model_dump()

                if not isinstance(content, dict):
                    continue

                tool_name = part.tool_name
                task_id = content.get("id")
                status = content.get("status")

                if not task_id:
                    continue

                # Track status updates
                if tool_name in ["images_create", "videos_create"]:
                    tasks[task_id] = status or "PENDING"
                elif tool_name in ["images_get", "videos_get"]:
                    tasks[task_id] = status

    # Filter for incomplete tasks
    # SUCCESS and FAILURE are final states
    pending = []
    for tid, status in tasks.items():
        if status not in ["SUCCESS", "FAILURE", "REVOKED", "REJECTED", "IGNORED"]:
            pending.append(tid)

    return pending

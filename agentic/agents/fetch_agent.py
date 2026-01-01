from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ToolReturnPart

from tools.dd_mcp_server import get_mcp_server

fetch_agent = Agent(
    "openai:gpt-4.1-mini",
    instructions=(
        "You retrieve pending images and videos by calling get_pending_image_ids and get_pending_video_ids. "
        "For each image ID, call images_get. For each video ID, call videos_get. "
        "Provide details of model and prompt used when returning media."
    ),
    deps_type=list[ModelMessage],
    toolsets=[get_mcp_server()],
    retries=1,
)


def _get_pending_tasks(messages: list[ModelMessage], target_type: str) -> list[str]:
    """Helper to scan message history for pending tasks of a specific type."""
    tasks: dict[str, dict[str, str]] = {}

    for msg in messages:
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

                # Determine type based on tool name
                task_type = "unknown"
                if tool_name and "images" in tool_name:
                    task_type = "image"
                elif tool_name and "videos" in tool_name:
                    task_type = "video"

                # Track status updates
                if task_id not in tasks:
                    tasks[task_id] = {"status": "PENDING", "type": task_type}

                if status:
                    tasks[task_id]["status"] = status

                # Update type if we found a more specific one
                if task_type != "unknown":
                    tasks[task_id]["type"] = task_type

    # Filter for incomplete tasks of the target type
    pending = []
    for tid, info in tasks.items():
        if info["type"] == target_type and info["status"] not in [
            "SUCCESS",
            "FAILURE",
            "REVOKED",
            "REJECTED",
            "IGNORED",
        ]:
            pending.append(tid)

    return pending


@fetch_agent.tool
def get_pending_image_ids(ctx: RunContext[list[ModelMessage]]) -> list[str]:
    """
    Scans the message history for image generation tasks that are not yet successful or failed.
    Returns a list of image task IDs.
    """
    return _get_pending_tasks(ctx.deps, "image")


@fetch_agent.tool
def get_pending_video_ids(ctx: RunContext[list[ModelMessage]]) -> list[str]:
    """
    Scans the message history for video generation tasks that are not yet successful or failed.
    Returns a list of video task IDs.
    """
    return _get_pending_tasks(ctx.deps, "video")

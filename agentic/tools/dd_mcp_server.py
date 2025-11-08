from __future__ import annotations as _annotations

import base64
import os
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.mcp import CallToolFunc, MCPServerStreamableHTTP, ToolResult

from common.logger import logger
from utils.utils import CACHE_DIR

# Create a global MCP server instance to avoid repeated connections
_mcp_server = None


def handle_image_get(data: dict) -> dict:
    """Handle image-specific processing"""
    if (
        "result" in data
        and isinstance(data["result"], dict)
        and "base64_data" in data["result"]
        and isinstance(data["result"]["base64_data"], str)
        and len(data["result"]["base64_data"]) > 1000
    ):
        try:
            # Decode and save base64
            image_data = base64.b64decode(data["result"]["base64_data"])

            # Generate filename from tool call info
            task_id = data.get("id", "unknown")
            filename = f"image_{task_id}.png"
            file_path = CACHE_DIR / filename

            # Save to file
            with open(file_path, "wb") as f:
                f.write(image_data)

            # Create modified response
            modified_data = data.copy()
            modified_result = data["result"].copy()
            modified_result["base64_data"] = f"[Image saved to: {file_path}]"
            modified_result["local_file_path"] = str(file_path)
            modified_result["file_size_kb"] = len(image_data) // 1024

            modified_data["result"] = modified_result

            logger.info(f"Cached image base64 ({len(data['result']['base64_data'])} chars) to {file_path}")
            return modified_data

        except Exception as e:
            logger.error(f"Failed to process image base64: {e}")
            # Return truncated version on error
            modified_data = data.copy()
            modified_result = data["result"].copy()
            modified_result["base64_data"] = (
                f"[Image base64 data truncated - {len(data['result']['base64_data'])} chars]"
            )
            modified_data["result"] = modified_result
            return modified_data

    return data


def handle_video_get(data: dict) -> dict:
    """Handle video-specific processing"""
    if (
        "result" in data
        and isinstance(data["result"], dict)
        and "base64_data" in data["result"]
        and isinstance(data["result"]["base64_data"], str)
        and len(data["result"]["base64_data"]) > 1000
    ):
        try:
            # Decode and save base64
            video_data = base64.b64decode(data["result"]["base64_data"])

            # Generate filename from tool call info
            task_id = data.get("id", "unknown")
            filename = f"video_{task_id}.mp4"
            file_path = CACHE_DIR / filename

            # Save to file
            with open(file_path, "wb") as f:
                f.write(video_data)

            # Create modified response
            modified_data = data.copy()
            modified_result = data["result"].copy()
            modified_result["base64_data"] = f"[Video saved to: {file_path}]"
            modified_result["local_file_path"] = str(file_path)
            modified_result["file_size_kb"] = len(video_data) // 1024

            modified_data["result"] = modified_result

            logger.info(f"Cached video base64 ({len(data['result']['base64_data'])} chars) to {file_path}")
            return modified_data

        except Exception as e:
            logger.error(f"Failed to process video base64: {e}")
            # Return truncated version on error
            modified_data = data.copy()
            modified_result = data["result"].copy()
            modified_result["base64_data"] = (
                f"[Video base64 data truncated - {len(data['result']['base64_data'])} chars]"
            )
            modified_data["result"] = modified_result
            return modified_data

    return data


async def process_tool_call(
    ctx: RunContext[Any],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Process tool calls to filter large base64 responses"""

    # Call the original tool
    result = await call_tool(name, tool_args, {"deps": ctx.deps})

    # Handle case where data might be a JSON string
    data = result
    logger.info(f"Processing tool call: {name}, {tool_args}, {type(data)}")
    logger.info(f"result: {data}")

    # Now data should be a dict (either originally or after JSON parsing)
    if name == "images_get":
        if not isinstance(data, dict):
            logger.warning(f"Tool call data is not a dict: {type(data)}")
            return result

        return handle_image_get(data)
    elif name == "videos_get":
        if not isinstance(data, dict):
            logger.warning(f"Tool call data is not a dict: {type(data)}")
            return result

        return handle_video_get(data)

    # Return original result if no processing needed
    return result


def get_mcp_server():
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServerStreamableHTTP(
            "http://localhost:5001/mcp",
            headers={"Authorization": f"Bearer {os.getenv('DDIFFUSION_API_KEY')}"},
            process_tool_call=process_tool_call,
        )
    return _mcp_server

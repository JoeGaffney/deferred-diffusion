"""Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent — the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:

    uv run -m pydantic_ai_examples.weather_agent
"""

from __future__ import annotations as _annotations

import asyncio
import base64
import os
from dataclasses import dataclass
from typing import Any

# from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPServerStreamableHTTP, ToolResult

from utils.utils import CACHE_DIR

# Create a global MCP server instance to avoid repeated connections
_mcp_server = None


async def process_tool_call(
    ctx: RunContext[Deps],
    call_tool: CallToolFunc,
    name: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Process tool calls to filter large base64 responses"""

    # Call the original tool
    result = await call_tool(name, tool_args, {"deps": ctx.deps})

    # Handle case where data might be a JSON string
    data = result
    print("Processing tool call:", name, tool_args, type(data))
    print("result:", data)

    # Now data should be a dict (either originally or after JSON parsing)
    if isinstance(data, dict):
        print("Tool call data is a dict with keys:", list(data.keys()))

        # Check if there's a nested 'result' with base64_data
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
                task_id = tool_args.get("id", "unknown")
                filename = f"{name}_{task_id}.png"
                file_path = CACHE_DIR / filename

                # Save to file
                with open(file_path, "wb") as f:
                    f.write(image_data)

                # Create modified response - modify the nested result
                modified_data = data.copy()
                modified_result = data["result"].copy()
                modified_result["base64_data"] = f"[Image saved to: {file_path}]"
                modified_result["local_file_path"] = str(file_path)
                modified_result["file_size_kb"] = len(image_data) // 1024

                modified_data["result"] = modified_result

                print(f"Cached large base64 ({len(data['result']['base64_data'])} chars) to {file_path}")

                # Return modified dict
                return modified_data

            except Exception as e:
                print(f"Failed to process base64: {e}")
                # Return truncated version on error
                modified_data = data.copy()
                modified_result = data["result"].copy()
                modified_result["base64_data"] = (
                    f"[Base64 data truncated - {len(data['result']['base64_data'])} chars]"
                )
                modified_data["result"] = modified_result

                return modified_data
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


@dataclass
class Deps:
    client: None


chat_agent = Agent(
    "openai:gpt-4.1-mini",
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions="You help the user with image generation requests. When creating images or videos just create them and return the task ids",
    deps_type=Deps,
    toolsets=[get_mcp_server()],
    retries=1,
)


# class LatLng(BaseModel):
#     lat: float
#     lng: float


# @chat_agent.tool
# async def get_lat_lng(ctx: RunContext[Deps], location_description: str) -> LatLng:
#     """Get the latitude and longitude of a location.

#     Args:
#         ctx: The context.
#         location_description: A description of a location.
#     """
#     # NOTE: the response here will be random, and is not related to the location description.
#     r = await ctx.deps.client.get(
#         "https://demo-endpoints.pydantic.workers.dev/latlng",
#         params={"location": location_description},
#     )
#     r.raise_for_status()
#     return LatLng.model_validate_json(r.content)


# @chat_agent.tool
# async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
#     """Get the weather at a location.

#     Args:
#         ctx: The context.
#         lat: Latitude of the location.
#         lng: Longitude of the location.
#     """
#     # NOTE: the responses here will be random, and are not related to the lat and lng.
#     temp_response, descr_response = await asyncio.gather(
#         ctx.deps.client.get(
#             "https://demo-endpoints.pydantic.workers.dev/number",
#             params={"min": 10, "max": 30},
#         ),
#         ctx.deps.client.get(
#             "https://demo-endpoints.pydantic.workers.dev/weather",
#             params={"lat": lat, "lng": lng},
#         ),
#     )
#     temp_response.raise_for_status()
#     descr_response.raise_for_status()
#     return {
#         "temperature": f"{temp_response.text} °C",
#         "description": descr_response.text,
#     }


async def main():
    deps = Deps(client=None)
    result = await chat_agent.run("What image models are available?", deps=deps)
    print("Response:", result.output)


if __name__ == "__main__":
    asyncio.run(main())

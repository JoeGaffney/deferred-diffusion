"""Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent — the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

Run with:

    uv run -m pydantic_ai_examples.weather_agent
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import logfire

# from httpx import AsyncClient
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import MCPServerStreamableHTTP

# Create a global MCP server instance to avoid repeated connections
_mcp_server = None


def get_mcp_server():
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServerStreamableHTTP(
            "http://localhost:5001/mcp", headers={"Authorization": f"Bearer {os.getenv('DDIFFUSION_API_KEY')}"}
        )
    return _mcp_server


@dataclass
class Deps:
    client: None


chat_agent = Agent(
    "openai:gpt-5-mini",
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    instructions="You help the user with image generation requests. When creating images or videos just create them and return the task ids",
    deps_type=Deps,
    toolsets=[get_mcp_server()],
    retries=0,
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

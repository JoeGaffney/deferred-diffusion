import pprint
from dataclasses import dataclass
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class ScriptDatabase:
    """This is a fake database for example purposes.
    In reality, you'd connect to a database with scene/script information.
    """

    @classmethod
    async def get_scene_context(cls, *, scene_id: int) -> str | None:
        if scene_id == 1:
            return "Science fiction setting on a space station"
        return None


@dataclass
class ScriptDependencies:
    scene_id: int
    db: ScriptDatabase


class ShotDescription(BaseModel):
    image_description: str = Field(description="Detailed visual description of the shot")
    camera_movement: str = Field(description="Description of camera movement and angles")
    dialog: str | None = Field(description="Any dialog or voice over in this shot", default=None)


class ScriptResult(BaseModel):
    shots: List[ShotDescription] = Field(description="Sequence of shots in the scene")
    mood: str = Field(description="Overall mood and atmosphere of the sequence")
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)


script_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=ScriptDependencies,
    result_type=ScriptResult,
    system_prompt=(
        "You are a professional storyboard artist and script supervisor. "
        "Break down scenes into detailed shot descriptions, including "
        "camera movements and dialog where appropriate."
    ),
)


@script_agent.system_prompt
async def add_scene_context(ctx: RunContext[ScriptDependencies]) -> str:
    scene_context = await ctx.deps.db.get_scene_context(scene_id=ctx.deps.scene_id)
    return f"Scene context: {scene_context}"


if __name__ == "__main__":
    deps = ScriptDependencies(scene_id=1, db=ScriptDatabase())
    result = script_agent.run_sync(
        "Create a sequence showing a person floating through a zero gravity corridor", deps=deps
    )
    pprint.pprint(result.data.model_dump(), indent=2)
    """
    shots=[
        ShotDescription(
            image_description='Wide shot of a sleek, metallic corridor with soft blue lighting. An astronaut in a casual space suit floating in the distance.',
            camera_movement='Slow, steady push-in following the floating figure',
            dialog=None
        ),
        ShotDescription(
            image_description='Medium shot of the astronaut gracefully navigating past floating equipment and papers',
            camera_movement='Tracking shot, slight dutch angle',
            dialog='*soft breathing sounds*'
        ),
        ShotDescription(
            image_description='Close-up of astronaut's face through helmet visor, showing peaceful expression',
            camera_movement='Static shot, shallow depth of field',
            dialog='Beautiful day for a float.'
        )
    ]
    mood='Serene and contemplative with a hint of wonder'
    duration_seconds=15
    """

import os
import pprint
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from common.logger import log_pretty, logger
from common.schemas import SequenceRequest, SequenceResponse
from tools.image_reference import main as image_reference_main

character_prompt = (
    "Generate a detailed visual description of the Person in the image. Ignore explaining the background. Focusing more on what the person looks like age, race, hair colour, what they may do for a job, there mood etc. To be used for storyboarding by a script agent. As a Characther description.",
)


class SequenceDatabase:
    """This is a fake database for example purposes.
    In reality, you'd connect to a database with scene/script information.
    """

    @classmethod
    async def get_scene_context(cls, *, scene_id: int) -> str | None:
        if scene_id == 1:
            return "Science fiction setting on a space station"
        return None


@dataclass
class SequenceDependencies:
    scene_id: int
    db: SequenceDatabase
    data: SequenceRequest


def get_agent():
    sequence_agent = Agent(
        "openai:gpt-5-mini",
        deps_type=SequenceDependencies,
        output_type=SequenceResponse,
        system_prompt=(
            "You are a professional storyboard artist and script supervisor. "
            "Break down scenes into detailed shot descriptions, including "
            "camera movements and dialog where appropriate."
            "use the add_scene_reference tool to get a reference for the scene. "
            "use the add_protaonist_reference tool to get a reference for the protagonist. "
            "use the add_antagonist_reference tool to get a reference for the antagonist. "
        ),
    )

    @sequence_agent.system_prompt
    async def add_scene_context(ctx: RunContext[SequenceDependencies]) -> str:
        scene_context = await ctx.deps.db.get_scene_context(scene_id=ctx.deps.scene_id)
        if scene_context == None:
            return ""

        return f"Scene context: {scene_context}"

    @sequence_agent.tool
    async def add_scene_reference(ctx: RunContext[SequenceDependencies]) -> str:
        if ctx.deps.data.scene_reference_image == None:
            return ""

        result = await image_reference_main(
            prompt="Generate a detailed visual description of the scene in the image. To be used for storyboarding by a script agent.",
            image_reference_image=ctx.deps.data.scene_reference_image,
        )

        if result == "":
            return ""
        return f"Scene context: {result}"

    @sequence_agent.tool
    async def add_protagonist_reference(ctx: RunContext[SequenceDependencies]) -> str:
        if ctx.deps.data.protagonist_reference_image == None:
            return ""

        result = await image_reference_main(
            prompt=str(character_prompt),
            image_reference_image=ctx.deps.data.protagonist_reference_image,
        )
        if result == "":
            return ""
        return f"Protagonist context: {result}"

    @sequence_agent.tool
    async def add_antagonist_reference(ctx: RunContext[SequenceDependencies]) -> str:
        if ctx.deps.data.antagonist_reference_image == None:
            return ""

        result = await image_reference_main(
            prompt=str(character_prompt),
            image_reference_image=ctx.deps.data.antagonist_reference_image,
        )
        if result == "":
            return ""
        return f"Antagonist context: {result}"

    return sequence_agent


def main(request: SequenceRequest) -> SequenceResponse:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    sequence_agent = get_agent()
    deps = SequenceDependencies(scene_id=2, db=SequenceDatabase(), data=request)
    result = sequence_agent.run_sync(request.prompt, deps=deps)
    history = result.all_messages()

    if request.refinement_prompt != "":
        result = sequence_agent.run_sync(request.refinement_prompt, deps=deps, message_history=history)
        history = result.all_messages()

    log_pretty("History", history)
    return result.output

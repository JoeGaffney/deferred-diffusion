import pprint
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from agentic.schemas import SequenceRequest, SequenceResponse
from agentic.tools.image_reference import main as image_reference_main
from utils.logger import log_pretty, logger


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


sequence_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=SequenceDependencies,
    result_type=SequenceResponse,
    system_prompt=(
        "You are a professional storyboard artist and script supervisor. "
        "Break down scenes into detailed shot descriptions, including "
        "camera movements and dialog where appropriate."
        "use the add_scene_reference tool to get a reference for the scene. "
        "use the add_protaonist_reference tool to get a reference for the protagonist. "
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

    result = image_reference_main(
        prompt="Generate a detailed visual description of the scene in the image. To be used for storyboarding by a script agent.",
        image_reference_image=ctx.deps.data.scene_reference_image,
    )
    if result == "":
        return ""
    return f"Scene context: {result}"


@sequence_agent.tool
def add_protagonist_reference(ctx: RunContext[SequenceDependencies]) -> str:
    if ctx.deps.data.protagonist_reference_image == None:
        return ""

    result = image_reference_main(
        prompt="Generate a detailed visual description of the protagonist in the image. To be used for storyboarding by a script agent.",
        image_reference_image=ctx.deps.data.protagonist_reference_image,
    )
    if result == "":
        return ""
    return f"Protagonist context: {result}"


def main(request: SequenceRequest) -> SequenceResponse:
    deps = SequenceDependencies(scene_id=2, db=SequenceDatabase(), data=request)
    result = sequence_agent.run_sync(request.prompt, deps=deps)
    history = result.all_messages()
    log_pretty("History stage 1", history)

    result = sequence_agent.run_sync(
        "Thats a good start can it be improved and all shot image_descriptions should be suitable for diffusion image prompts"
        "and include the set description",
        deps=deps,
        message_history=history,
    )
    history = result.all_messages()

    log_pretty("History", history)
    log_pretty("Result", result.data.model_dump())
    return result.data


if __name__ == "__main__":
    main(
        SequenceRequest(
            prompt="Create a sequence about a man on an adventure.",
            scene_reference_image="../test_data/color_v001.jpeg",
            protagonist_reference_image="../test_data/face_v001.jpeg",
        )
    )

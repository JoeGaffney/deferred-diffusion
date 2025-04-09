import pprint
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from agentic.schemas import SequenceAgentRequest, SequenceAgentResponse


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


sequence_agent = Agent(
    "openai:gpt-4o-mini",
    deps_type=SequenceDependencies,
    result_type=SequenceAgentResponse,
    system_prompt=(
        "You are a professional storyboard artist and script supervisor. "
        "Break down scenes into detailed shot descriptions, including "
        "camera movements and dialog where appropriate."
    ),
)


@sequence_agent.system_prompt
async def add_scene_context(ctx: RunContext[SequenceDependencies]) -> str:
    scene_context = await ctx.deps.db.get_scene_context(scene_id=ctx.deps.scene_id)
    return f"Scene context: {scene_context}"


def main(request: SequenceAgentRequest) -> SequenceAgentResponse:
    deps = SequenceDependencies(scene_id=2, db=SequenceDatabase())
    result = sequence_agent.run_sync(request.prompt, deps=deps)
    pprint.pprint(result.data.model_dump(), indent=2)
    return result.data


if __name__ == "__main__":
    main(SequenceAgentRequest(prompt="Create a sequence showing a person floating through a zero gravity corridor"))

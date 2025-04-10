from typing import List

from pydantic import BaseModel, Field


class ShotResponse(BaseModel):
    name: str = Field(description="Unique identifier name for the shot eg 001, 002")
    image_description: str = Field(description="Detailed visual description of the shot")
    camera_movement: str = Field(description="Description of camera movement and angles")
    dialog: str | None = Field(description="Any dialog or voice over in this shot", default=None)
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)


class CharacterResponse(BaseModel):
    name: str = Field(description="Name of the character")
    role: str = Field(description="Role of the character in the scene")
    emotion: str = Field(description="Emotion or state of the character")
    image_description: str = Field(description="Visual description of the character")


class SceneResponse(BaseModel):
    name: int = Field(description="Name of thes scene eg. castle, forest, interior_kitchen")
    sequence_mame: str = Field(description="Unique identifier Title of the scene")
    image_description: str = Field(description="Detailed description of the scene")
    location: str = Field(description="Location where the scene takes place")
    time_of_day: str = Field(description="Time of day for the scene")
    mood: str = Field(description="Overall mood and atmosphere of the sequence")
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)
    diffusion_postive_prompt_tags: str = Field(
        description="Diffusion positive prompt tags eg. realistic, 4k, DLSR photo etc."
    )
    diffusion_negative_prompt_tags: str = Field(
        description="Diffusion negative prompt tags eg. blurry, lowres, bad anatomy etc."
    )


class SequenceRequest(BaseModel):
    prompt: str = ""


class SequenceResponse(BaseModel):
    scene: SceneResponse = Field(description="Description of the scene")
    characters: List[CharacterResponse] = Field(description="List of characters in the scene")
    shots: List[ShotResponse] = Field(description="Sequence of shots in the scene")

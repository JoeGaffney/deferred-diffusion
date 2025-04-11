from typing import List

from pydantic import BaseModel, Field


class ShotCharacterResponse(BaseModel):
    name: str = Field(description="Name of the character")
    action: str = Field(description="Action of the character in this shot")
    dialog: str | None = Field(description="Dialog of the character in this shot", default=None)
    frame_position: str = Field(description="Position of the character in the frame eg. left, right, center")


class ShotResponse(BaseModel):
    name: str = Field(description="Unique identifier name for the shot eg 001, 002")
    antagonist: ShotCharacterResponse | None = Field(
        description="Description of the antagonist's action and dialog in this shot"
    )
    protagonist: ShotCharacterResponse | None = Field(
        description="Description of the protagonist's action and dialog in this shot"
    )
    camera_movement: str = Field(description="Description of camera movement and angles")
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)
    image_description: str = Field(description="Detailed visual description of the shot")


class CharacterResponse(BaseModel):
    name: str = Field(description="Name of the character")
    role: str = Field(description="Role of the character in the scene")
    emotion: str = Field(description="Emotion or state of the character")
    image_description: str = Field(description="Visual description of the character")
    image_portrait_description: str = Field(
        description="Visual description of the character's portrait or close-up shot, photo shoot style neautral background"
    )


class SceneResponse(BaseModel):
    name: str = Field(description="Name of thes scene eg. castle, forest, interior_kitchen")
    mood: str = Field(description="Overall mood and atmosphere of the sequence")
    location: str = Field(description="Location of the scene")
    time_of_day: str = Field(description="Time of day for the theme style")
    image_description: str = Field(description="Detailed description of the scene")
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
    protagonist: CharacterResponse | None = Field(description="Description of the protagonist")
    antagonist: CharacterResponse | None = Field(description="Description of the antagonist")
    shots: List[ShotResponse] = Field(description="Sequence of shots in the scene")

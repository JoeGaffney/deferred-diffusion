from typing import List

from pydantic import BaseModel, Field


class ShotDescription(BaseModel):
    image_description: str = Field(description="Detailed visual description of the shot")
    camera_movement: str = Field(description="Description of camera movement and angles")
    dialog: str | None = Field(description="Any dialog or voice over in this shot", default=None)
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)


class CharacterDescription(BaseModel):
    name: str = Field(description="Name of the character")
    role: str = Field(description="Role of the character in the scene")
    emotion: str = Field(description="Emotion or state of the character")
    image_description: str = Field(description="Visual description of the character")


class SceneDescription(BaseModel):
    scene_id: int = Field(description="Unique identifier for the scene")
    title: str = Field(description="Title of the scene")
    image_description: str = Field(description="Detailed description of the scene")
    location: str = Field(description="Location where the scene takes place")
    time_of_day: str = Field(description="Time of day for the scene")


class ScriptAgentRequest(BaseModel):
    prompt: str = ""


class ScriptAgentResponse(BaseModel):
    shots: List[ShotDescription] = Field(description="Sequence of shots in the scene")
    mood: str = Field(description="Overall mood and atmosphere of the sequence")
    scene: SceneDescription = Field(description="Description of the scene")
    characters: List[CharacterDescription] = Field(description="List of characters in the scene")
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)

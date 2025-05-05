from typing import List

from pydantic import Base64Bytes, BaseModel, Field, computed_field, field_serializer


class ShotCharacterResponse(BaseModel):
    # name: str = Field(description="Name of the character")
    gender: str = Field(description="Gender of the character")
    action: str = Field(description="Action of the character in this shot")
    dialog: str | None = Field(description="Dialog of the character in this shot", default=None)
    frame_position: str = Field(description="Position of the character in the frame eg. left, right, center")


class SceneResponse(BaseModel):
    name: str = Field(description="Short name of the scene, e.g., 'forest', 'interior_kitchen'")
    mood: str = Field(description="Overall mood and atmosphere, e.g., 'mysterious', 'cozy', 'tense'")
    location: str = Field(description="Specific or general location, e.g., 'mountaintop temple', 'urban alleyway'")
    time_of_day: str = Field(description="Time of day, e.g., 'sunset', 'night', 'early morning'")
    weather: str = Field(description="Weather or atmospheric conditions, e.g., 'foggy', 'raining', 'clear skies'")
    lighting: str = Field(description="Lighting setup, e.g., 'natural sunlight', 'low light', 'dramatic shadows'")
    diffusion_positive_prompt_tags: str = Field(
        description="List of prompt tags for image generation, e.g., '4k, DLSR photo, ultra-realistic, soft lighting, cinematic'"
    )

    @computed_field
    @property
    def image_prompt(self) -> str:
        return (
            f"A {self.mood} scene set in a {self.location} during {self.time_of_day}, "
            f"The atmosphere is {self.weather}, with {self.lighting}, {self.diffusion_positive_prompt_tags}"
        )


class ShotResponse(BaseModel):
    name: str = Field(description="Unique identifier name for the shot eg 001, 002")
    antagonist: ShotCharacterResponse | None = Field(
        description="Description of the antagonist's action and dialog in this shot"
    )
    protagonist: ShotCharacterResponse | None = Field(
        description="Description of the protagonist's action and dialog in this shot"
    )
    characters: str = Field(
        description="Description of the characters in the shot, e.g., 'protagonist character gender and antagonist character gender', 'crowd of people' including their actions and positions"
    )
    subject: str = Field(
        description="Description of the subject in the shot, eg. character, character gender. object, environment"
    )
    environment: str = Field(
        description="Description of the environment in the shot, eg. castle, forest, interior_kitchen"
    )
    lighting: str = Field(description="Description of the lighting in the shot, eg. bright, dark, moody")
    action: str = Field(description="Description of the action taking place in the shot, eg. fight, chase")
    style: str = Field(description="Description of the style of the shot, eg. realistic, cartoon, anime")
    camera_angle: str = Field(description="Description of the camera angle in the shot, eg. close-up, wide shot")
    camera_movement: str = Field(description="Description of camera movement and angles eg. pan, tilt, dolly, zoom")
    duration_seconds: int = Field(description="Estimated duration in seconds", ge=1, le=300)

    @computed_field
    @property
    def image_prompt(self) -> str:
        parts = [
            self.subject,
            self.characters,
            f"in {self.environment}",
            self.action,
            f"Style: {self.style}",
            f"Lighting: {self.lighting}",
            f"Camera angle: {self.camera_angle}",
        ]
        return ", ".join(part for part in parts if part)

    @computed_field
    @property
    def video_prompt(self) -> str:
        parts = [
            self.subject,
            f"in {self.environment}",
            self.action,
            f"Style: {self.style}",
            f"Lighting: {self.lighting}",
            f"Camera angle: {self.camera_angle}",
            f"Camera movement: {self.camera_movement}",
        ]
        return ", ".join(part for part in parts if part)


class CharacterResponse(BaseModel):
    name: str = Field(description="Name of the character")
    role: str = Field(description="Role of the character in the scene, e.g., protagonist, antagonist, sidekick")
    gender: str = Field(description="Gender identity of the character")
    age: str = Field(description="Approximate age or age range")
    ethnicity: str = Field(description="Ethnic background or appearance of the character")
    height: str = Field(description="Character's height or stature, e.g., tall, short, average")
    build: str = Field(description="Body build of the character, e.g., athletic, lean, bulky")
    eye_color: str = Field(description="Eye color or description")
    hair_color: str = Field(description="Hair color or style")
    profession: str = Field(description="Character's profession or occupation, if relevant")
    attire: str = Field(description="Clothing and accessories they wear")

    @computed_field
    @property
    def image_prompt(self) -> str:
        return (
            f"Portrait of {self.name}, a {self.age}-year-old {self.ethnicity} {self.gender}, "
            f"shot against a neutral background. They have {self.hair_color} hair, "
            f"{self.eye_color} eyes, and are wearing {self.attire}. Studio lighting, clean composition."
        )


class SequenceRequest(BaseModel):
    prompt: str = ""
    refinement_prompt: str = ""
    scene_reference_image: str | None = Field(description="Reference image for the scene", default=None)
    protagonist_reference_image: str | None = Field(description="Reference image for the protagonist", default=None)
    antagonist_reference_image: str | None = Field(description="Reference image for the antagonist", default=None)


class SequenceResponse(BaseModel):
    scene: SceneResponse = Field(description="Description of the scene")
    protagonist: CharacterResponse | None = Field(description="Description of the protagonist")
    antagonist: CharacterResponse | None = Field(description="Description of the antagonist")
    shots: List[ShotResponse] = Field(description="Sequence of shots in the scene")

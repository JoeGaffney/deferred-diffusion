from typing import Literal

from pydantic import BaseModel, Field


class Wan21Params(BaseModel):
    prompt: str = Field(..., description="Prompt for the video generation")
    seed: int = Field(default=-1)
    duration_seconds: int = Field(default=5)
    resolution: Literal["480p", "720p", "1080p"] = Field(default="480p")

    resolution: Literal["480p", "720p", "1080p"] = Field(default="480p")

    resolution: Literal["480p", "720p", "1080p"] = Field(default="480p")

    resolution: Literal["480p", "720p", "1080p"] = Field(default="480p")

    resolution: Literal["480p", "720p", "1080p"] = Field(default="480p")

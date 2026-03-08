from pydantic import BaseModel, Field


class Flux1Params(BaseModel):
    prompt: str = Field(..., description="The prompt to generate the image")
    seed: int = Field(default=-1, description="Random seed, -1 for random")
    width: int = Field(default=1024, description="Image width")
    height: int = Field(default=1024, description="Image height")

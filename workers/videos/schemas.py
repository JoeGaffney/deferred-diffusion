from typing import Literal

from pydantic import Base64Bytes, BaseModel, Field


class VideoRequest(BaseModel):
    model: Literal[
        "LTX-Video",
        "Wan2.1",
        "runway/gen3a_turbo",
        "runway/gen4_turbo",
    ]
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    prompt: str = "Slow camera zoom in, 4k, high quality, cinematic, realistic"
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    guidance_scale: float = 5.0
    num_frames: int = 48
    num_inference_steps: int = 25
    seed: int = 42


class VideoWorkerResponse(BaseModel):
    base64_data: Base64Bytes

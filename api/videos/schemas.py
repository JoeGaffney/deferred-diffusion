from typing import Literal, Optional
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field


class VideoRequest(BaseModel):
    model: Literal[
        "LTX-Video",
        "HunyuanVideo",
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


class VideoResponse(BaseModel):
    id: UUID
    status: str
    result: Optional[VideoWorkerResponse] = None
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
                },
                "error_message": None,
            }
        }


class VideoCreateResponse(BaseModel):
    id: UUID
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }

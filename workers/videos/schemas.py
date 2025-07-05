from typing import Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2-1",
    "external-runway-gen-3",
    "external-runway-gen-4",
]
ModelFamily: TypeAlias = Literal["ltx", "wan", "runway"]
TaskName: TypeAlias = Literal["process_video", "process_video_external"]


class ModelInfo(BaseModel):
    family: ModelFamily
    path: str
    external: bool


MODEL_CONFIG: Dict[ModelName, ModelInfo] = {
    "ltx-video": ModelInfo(family="ltx", path="Lightricks/LTX-Video-0.9.7-distilled", external=False),
    "wan-2-1": ModelInfo(family="wan", path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", external=False),
    "external-runway-gen-3": ModelInfo(family="runway", path="gen3a_turbo", external=True),
    "external-runway-gen-4": ModelInfo(family="runway", path="gen4_turbo", external=True),
}


class VideoRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Slow camera zoom in, 4k, high quality, cinematic, realistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
        description="Negative prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    guidance_scale: float = 5.0
    num_frames: int = 48
    num_inference_steps: int = 25
    seed: int = 42
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    image_last_frame: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string for the last frame",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_CONFIG[self.model].family

    @property
    def model_path(self) -> str:
        return MODEL_CONFIG[self.model].path

    @property
    def external_model(self) -> bool:
        return MODEL_CONFIG[self.model].external

    @property
    def task_name(self) -> TaskName:
        """Determines the appropriate task name based on request characteristics."""
        if self.external_model:
            return "process_video_external"
        return "process_video"

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return "cpu" if self.external_model else "gpu"


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


class VideoModelInfoResponse(BaseModel):
    name: ModelName
    family: ModelFamily
    path: str
    external: bool

from typing import Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelName: TypeAlias = Literal[
    "LTX-Video",
    "Wan2.1",
    "runway/gen3a_turbo",
    "runway/gen4_turbo",
]
ModelFamily: TypeAlias = Literal["ltx", "wan", "runway"]
TaskName: TypeAlias = Literal["process_video", "process_video_workflow", "process_video_external"]


class ComfyWorkflow(BaseModel):
    """Represents a ComfyUI workflow with dynamic node structure."""

    model_config = {
        "extra": "allow",
    }


class VideoRequest(BaseModel):
    model: ModelName
    comfy_workflow: Optional[ComfyWorkflow] = Field(
        default=None,
        description="ComfyUI workflow configuration with dynamic node structure",
    )
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

    @property
    def model_family(self) -> ModelFamily:
        mapping: Dict[ModelName, ModelFamily] = {
            "LTX-Video": "ltx",
            "Wan2.1": "wan",
            "runway/gen3a_turbo": "runway",
            "runway/gen4_turbo": "runway",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model family defined for model '{self.model}'")

    @property
    def model_path(self) -> str:
        mapping: Dict[ModelName, str] = {
            "LTX-Video": "Lightricks/LTX-Video-0.9.7-distilled",
            "Wan2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            "runway/gen3a_turbo": "gen3a_turbo",
            "runway/gen4_turbo": "gen4_turbo",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model path defined for model '{self.model}'")

    @property
    def external_model(self) -> bool:
        external_models = ["runway"]
        return self.model_family in external_models

    @property
    def task_name(self) -> TaskName:
        """Determines the appropriate task name based on request characteristics."""
        if self.comfy_workflow:
            return "process_video_workflow"
        if self.external_model:
            return "process_video_external"
        return "process_video"


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

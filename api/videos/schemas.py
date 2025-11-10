from typing import Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, model_validator

# User facing choice
ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2",
    "runway-gen-4",
    "runway-act-two",
    "runway-upscale",
    "runway-gen-4-aleph",
    "bytedance-seedance-1",
    "kwaivgi-kling-2",
    "google-veo-3",
    "openai-sora-2",
    "minimax-hailuo-2",
]

InferredMode: TypeAlias = Literal["text-to-video", "image-to-video", "video-to-video", "first-last-image"]
Provider: TypeAlias = Literal["local", "openai", "replicate", "runway"]


class VideosModelInfo(BaseModel):
    provider: Provider = Field(description="Source/provider identifier")
    external: bool = Field(description="True if the model is invoked via an external API")
    supported_modes: set[InferredMode] = Field(default_factory=set)
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"

    def supports_inferred_mode(self, mode: InferredMode) -> bool:
        return mode in self.supported_modes


# Unified metadata (local + external)
MODEL_META: Dict[ModelName, VideosModelInfo] = {
    # Local
    "ltx-video": VideosModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-video", "image-to-video", "first-last-image", "video-to-video"},
        description="Fast but more limited video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2": VideosModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-video", "image-to-video", "first-last-image", "video-to-video"},
        description="Wan 2.2, quality open-source video generation model. Will fall back to Wan VACE 2.1 for video-to-video.",
    ),
    # External
    "runway-gen-4": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"image-to-video"},
        description="Runway Gen-4 general video generation fast but limited.",
    ),
    "runway-act-two": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"video-to-video"},
        description="Matches animation from a reference video to a character reference image.",
    ),
    "runway-upscale": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"video-to-video"},
        description="Video upscaling model.",
    ),
    "runway-gen-4-aleph": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"video-to-video"},
        description="Aleph can enhance/alter existing video and use image references.",
    ),
    "bytedance-seedance-1": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-video", "image-to-video", "first-last-image"},
        description="Seedance-1 flagship model. Great all rounder. Supports high_quality variant.",
    ),
    "kwaivgi-kling-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-video", "image-to-video", "first-last-image"},
        description="Kling 2.5 flagship model. Great at first-last frame coherence.",
    ),
    "google-veo-3": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-video", "image-to-video", "first-last-image"},
        description="VEO-3.1 flagship model. Expensive.",
    ),
    "openai-sora-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-video", "image-to-video"},
        description="Sora 2 openai flagship model. Expensive and not great at image-to-video.",
    ),
    "minimax-hailuo-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-video", "image-to-video"},
        description="Hailuo-2.3 great physics understanding.",
    ),
}


def generate_model_docs():
    header = (
        "# Video Models\n"
        "External models proxy to provider APIs; local models run on your GPU.\n\n"
        "| Model | Provider | External | Queue | Modes | Description |\n"
        "|-------|----------|:--------:|:-----:|-------|-------------|\n"
    )
    rows = []
    for name, meta in MODEL_META.items():
        modes = ", ".join(sorted(meta.supported_modes))
        rows.append(
            f"| {name} | {meta.provider} | {'Yes' if meta.external else 'No'} | {meta.queue} | {modes} | {meta.description or ''} |"
        )
    return header + "\n".join(rows) + "\n"


class VideoRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Slow camera zoom in, 4k, high quality, cinematic, realistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    height: int = 480
    width: int = 854
    num_frames: int = 48
    seed: int = 42
    image: Optional[str] = Field(
        default=None,
        description="Base64 image string used for image-to-video conditioning or reference.",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    last_image: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string for the last frame guidance in image-to-video generation (requires image).",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    video: Optional[str] = Field(
        default=None,
        description="Optional Base64 video string for video-to-video transformation or upscaling.",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "video/*",
        },
    )
    high_quality: bool = Field(
        default=False,
        description="Use high quality model variant when available (may cost more and take longer). Will use higher steps in local models.",
    )

    @property
    def inferred_mode(self) -> InferredMode:
        if self.video:
            return "video-to-video"
        if self.image and self.last_image:
            return "first-last-image"
        if self.image:
            return "image-to-video"
        return "text-to-video"

    @property
    def meta(self) -> VideosModelInfo:
        return MODEL_META[self.model]

    @property
    def external_model(self) -> bool:
        return self.meta.external

    @property
    def task_name(self) -> ModelName:
        return self.model

    @property
    def task_queue(self) -> str:
        return self.meta.queue

    @model_validator(mode="after")
    def _validate_capabilities(self):
        mode = self.inferred_mode
        if not self.meta.supports_inferred_mode(mode):
            raise ValueError(f"Model '{self.model}' does not support mode '{mode}'.")
        if self.last_image and not self.image:
            raise ValueError("last_image requires image.")
        return self


class VideoWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class VideoResponse(BaseModel):
    id: UUID
    status: str
    result: Optional[VideoWorkerResponse] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
                },
                "error_message": None,
            }
        }
    )


class VideoCreateResponse(BaseModel):
    id: UUID
    status: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }
    )


class VideoModelsResponse(BaseModel):
    models: Dict[ModelName, VideosModelInfo]

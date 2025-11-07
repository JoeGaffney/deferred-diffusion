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

InferredMode: TypeAlias = Literal["text_to_video", "image_to_video", "video_to_video", "first_last_frame"]
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
        supported_modes={"text_to_video", "image_to_video", "first_last_frame", "video_to_video"},
        description="Fast but more limited video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2": VideosModelInfo(
        provider="local",
        external=False,
        supported_modes={"text_to_video", "image_to_video", "first_last_frame", "video_to_video"},
        description="Wan 2.2, quality open-source video generation model. Good motion coherence for varied scenes.",
    ),
    # External
    "runway-gen-4": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"text_to_video", "image_to_video"},
        description="Runway Gen-4 general video generation (text/image to video).",
    ),
    "runway-act-two": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"text_to_video", "image_to_video", "video_to_video"},
        description="Updates an input video with a reference image (image+video to video).",
    ),
    "runway-upscale": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"video_to_video"},
        description="Video upscaling model (video-to-video).",
    ),
    "runway-gen-4-aleph": VideosModelInfo(
        provider="runway",
        external=True,
        supported_modes={"text_to_video", "image_to_video", "first_last_frame", "video_to_video"},
        description="Aleph variant: can enhance/alter existing video and use image references.",
    ),
    "bytedance-seedance-1": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_video", "image_to_video"},
        description="Seedance-1 fast general video generation.",
    ),
    "kwaivgi-kling-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_video", "image_to_video"},
        description="Kling V2 high-quality text/image to video generation.",
    ),
    "google-veo-3": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_video", "image_to_video"},
        description="VEO-3 flagship model. Supports high_quality variant.",
    ),
    "openai-sora-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_video", "image_to_video"},
        description="Sora 2 high-end text/image to video generation. Supports high_quality variant.",
    ),
    "minimax-hailuo-2": VideosModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_video"},
        description="Hailuo-2.3 text-to-video with multiple duration/resolution options.",
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
    height: int = 720
    width: int = 1280
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
            return "video_to_video"
        if self.image and self.last_image:
            return "first_last_frame"
        if self.image:
            return "image_to_video"
        return "text_to_video"

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
        if self.video and not self.image:
            raise ValueError("video requires image.")
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

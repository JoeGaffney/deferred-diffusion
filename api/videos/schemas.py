from typing import Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from common.schemas import Base64Image, Base64Video, Provider, TaskStatus

# User facing choice
ModelName: TypeAlias = Literal[
    "ltx-video",
    "wan-2",
    "hunyuan-video-1",
    "sam-3",
    "runway-gen-4",
    "runway-upscale",
    "seedance-1",
    "kling-2",
    "veo-3",
    "sora-2",
    "hailuo-2",
]


class VideosModelInfo(BaseModel):
    external: bool = Field(description="True if the model is invoked via an external API")
    provider: Provider = Field(description="Source/provider identifier")
    text_to_video: bool = False
    image_to_video: bool = False
    video_to_video: bool = False
    last_image: bool = False
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"


# Unified metadata (local + external)
MODEL_META: Dict[ModelName, VideosModelInfo] = {
    "ltx-video": VideosModelInfo(
        provider="local",
        external=False,
        text_to_video=True,
        image_to_video=True,
        video_to_video=True,
        last_image=True,
        description="Fast but more limited video generation model. Good for quick iterations and less complex scenes.",
    ),
    "wan-2": VideosModelInfo(
        provider="local",
        external=False,
        text_to_video=True,
        image_to_video=True,
        video_to_video=True,
        last_image=True,
        description="Wan 2.2, quality open-source video generation model. Will fall back to Wan VACE 2.1 for video-to-video.",
    ),
    "hunyuan-video-1": VideosModelInfo(
        provider="local",
        external=False,
        text_to_video=True,
        image_to_video=True,
        description="Hunyuan Video 1.5.",
    ),
    "sam-3": VideosModelInfo(
        provider="local",
        external=False,
        video_to_video=True,
        description="SAM-3, segmentation model. (Broken atm)",
    ),
    "runway-gen-4": VideosModelInfo(
        provider="replicate",
        external=True,
        image_to_video=True,
        video_to_video=True,
        description="Runway Gen-4 family. Uses standard Gen-4 for image-to-video and Aleph variant for video-to-video.",
    ),
    "runway-upscale": VideosModelInfo(
        provider="replicate",
        external=True,
        video_to_video=True,
        description="Runway's video upscaling model.",
    ),
    "seedance-1": VideosModelInfo(
        provider="replicate",
        external=True,
        text_to_video=True,
        image_to_video=True,
        last_image=True,
        description="Bytedance Seedance-1 flagship model. Great all rounder.",
    ),
    "kling-2": VideosModelInfo(
        provider="replicate",
        external=True,
        text_to_video=True,
        image_to_video=True,
        last_image=True,
        description="Kling 2.5 flagship model. Great at first-last frame coherence.",
    ),
    "veo-3": VideosModelInfo(
        provider="replicate",
        external=True,
        text_to_video=True,
        image_to_video=True,
        last_image=True,
        description="Googles VEO-3.1 flagship model. Expensive.",
    ),
    "sora-2": VideosModelInfo(
        provider="replicate",
        external=True,
        text_to_video=True,
        image_to_video=True,
        description="OpenAI's Sora 2 openai flagship model. Expensive and not great at image-to-video.",
    ),
    "hailuo-2": VideosModelInfo(
        provider="replicate",
        external=True,
        text_to_video=True,
        image_to_video=True,
        description="Minimax's Hailuo-2.3 great physics understanding.",
    ),
}


def generate_model_docs():
    header = "# Video Models\n" "External models proxy to provider APIs; local models run on your GPU.\n\n"

    # Get all fields from VideosModelInfo to build table header
    model_fields = VideosModelInfo.model_fields
    columns = ["Model"] + [field_name.replace("_", " ").title() for field_name in model_fields.keys()]

    # Build header row
    header += "| " + " | ".join(columns) + " |\n"
    header += "|" + "|".join([":-------:" if i > 0 else "-------" for i in range(len(columns))]) + "|\n"

    rows = []
    for name, meta in MODEL_META.items():
        row_values = [name]
        for field_name, field_info in model_fields.items():
            value = getattr(meta, field_name)

            # Format based on type
            if isinstance(value, bool):
                formatted = "✓" if value else "✗"
            elif value is None:
                formatted = ""
            else:
                formatted = str(value)

            row_values.append(formatted)

        rows.append("| " + " | ".join(row_values) + " |")

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
    num_frames: int = Field(
        default=48,
        description="Preferred number of frames to generate. External models will round to the nearest supported duration (e.g., 5s or 10s intervals). Values above 100 frames will automatically use the next available duration range.",
        ge=24,
        le=250,
    )
    seed: int = 42
    image: Optional[Base64Image] = Field(
        default=None,
        description="Base64 image string used for image-to-video conditioning or reference.",
    )
    last_image: Optional[Base64Image] = Field(
        default=None,
        description="Optional Base64 image string for the last frame guidance in image-to-video generation (requires image).",
    )
    video: Optional[Base64Video] = Field(
        default=None,
        description="Optional Base64 video string for video-to-video transformation or upscaling.",
    )

    @property
    def meta(self) -> VideosModelInfo:
        return MODEL_META[self.model]

    @property
    def external_model(self) -> bool:
        return self.meta.external

    @property
    def task_name(self) -> str:
        return f"videos.{self.model}"

    @property
    def task_queue(self) -> str:
        return self.meta.queue

    @property
    def cleaned_prompt(self) -> str:
        return " ".join(self.prompt.split())

    @model_validator(mode="after")
    def _validate_capabilities(self):
        if self.video:
            # Video-to-video mode
            if not self.meta.video_to_video:
                raise ValueError(f"Model '{self.model}' does not support video_to_video.")
        elif self.image:
            # Image-to-video mode
            if not self.meta.image_to_video:
                raise ValueError(f"Model '{self.model}' does not support image_to_video.")
        else:
            # Text-to-video mode
            if not self.meta.text_to_video:
                raise ValueError(f"Model '{self.model}' does not support text_to_video.")
        if self.last_image and not self.image:
            raise ValueError("last_image requires image.")
        if self.last_image and not self.meta.last_image:
            raise ValueError(f"Model '{self.model}' does not support last_image capability.")
        return self


class VideoWorkerResponse(BaseModel):
    output: List[str]
    logs: List[str]


class VideoResponse(BaseModel):
    id: UUID
    status: TaskStatus
    output: List[HttpUrl] = []
    error_message: Optional[str] = None
    logs: List[str] = []
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "output": ["http://localhost:5000/api/files/..."],
                "error_message": None,
                "logs": ["Setup", "Progress: 10%", "Progress: 20%", "..."],
            }
        }
    )


class VideoCreateResponse(BaseModel):
    id: UUID
    status: TaskStatus
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

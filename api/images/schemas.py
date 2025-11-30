from typing import Any, Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, model_validator

from common.schemas import Provider, TaskStatus

# User facing choice
ModelName: TypeAlias = Literal[
    "sd-xl",
    "flux-1",
    "flux-2",
    "qwen-image",
    "z-image",
    "depth-anything-2",
    "sam-2",
    "sam-3",
    "real-esrgan-x4",
    "gpt-image-1",
    "runway-gen4-image",
    "flux-1-pro",
    "flux-2-pro",
    "topazlabs-upscale",
    "google-gemini-2",
    "google-gemini-3",
    "bytedance-seedream-4",
]
InferredMode: TypeAlias = Literal["text-to-image", "image-to-image", "inpainting"]


class ImagesModelInfo(BaseModel):
    provider: Provider = Field(description="Source/provider identifier")
    external: bool = Field(description="True if the model is invoked via an external API")
    supported_modes: set[InferredMode] = Field(default_factory=set)
    references: bool = False  # separate capability flag
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"

    def supports_inferred_mode(self, mode: InferredMode) -> bool:
        return mode in self.supported_modes


# Unified metadata
MODEL_META: Dict[ModelName, ImagesModelInfo] = {
    "sd-xl": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        references=True,
        description="Stable Diffusion XL variant with broad adapter/controlnet support.",
    ),
    "flux-1": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        references=True,
        description="FLUX dev model (Krea tuned). Uses Kontext for image-to-image, Fill for inpainting.",
    ),
    "flux-2": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        references=True,
        description="FLUX 2.0 dev model with edit capabilities.",
    ),
    "qwen-image": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        references=True,
        description="Qwen image generation and manipulation.",
    ),
    "z-image": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text-to-image"},
        description="Z-Image open-source image generation model.",
    ),
    "depth-anything-2": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"image-to-image"},
        description="Depth estimation pipeline.",
    ),
    "sam-2": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"image-to-image"},
        description="Meta's SAM 2 Segmentation pipeline.",
    ),
    "sam-3": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"image-to-image"},
        description="Meta's SAM 3 Segmentation pipeline.",
    ),
    "gpt-image-1": ImagesModelInfo(
        provider="openai",
        external=True,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        references=True,
        description="OpenAI image model.",
    ),
    "runway-gen4-image": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image"},
        references=True,
        description="Runway Gen-4 image model.",
    ),
    "flux-1-pro": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        description="FLUX 1.1 Pro variants via external provider.",
    ),
    "flux-2-pro": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image", "inpainting"},
        description="FLUX 2.0 Pro variants via external provider.",
        references=True,
    ),
    "topazlabs-upscale": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"image-to-image"},
        description="Topaz upscale model.",
    ),
    "google-gemini-2": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image"},
        references=True,
        description="Gemini multimodal image model.",
    ),
    "google-gemini-3": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image"},
        references=True,
        description="Gemini 3 Pro image model.",
    ),
    "bytedance-seedream-4": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text-to-image", "image-to-image"},
        references=True,
        description="Seedream image-to-image context model.",
    ),
}


def generate_model_docs():
    header = (
        "# Image Models\n"
        "External models proxy to provider APIs; local models run on your GPU.\n\n"
        "| Model | Provider | External | Queue | Modes | References | Description |\n"
        "|-------|----------|:--------:|:-----:|-------|:----------:|-------------|\n"
    )
    rows = []
    for name, meta in MODEL_META.items():
        modes = ", ".join(sorted(meta.supported_modes))
        rows.append(
            f"| {name} | {meta.provider} | {'Yes' if meta.external else 'No'} | {meta.queue} | {modes} | {'✓' if meta.references else '✗'} | {meta.description or ''} |"
        )
    return header + "\n".join(rows) + "\n"


class References(BaseModel):
    mode: Literal["style", "face", "depth", "canny", "pose"]
    strength: float = 0.5
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    mask: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )


class ImageRequest(BaseModel):
    model: ModelName
    prompt: str = Field(
        default="Detailed, 8k, photorealistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    height: int = 720
    width: int = 1280
    seed: int = 42
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strongly to follow the input image when transforming it (image-to-image/inpainting only). Ignored for text-to-image.",
    )
    image: Optional[str] = Field(
        default=None,
        description=(
            "Base64 string image. If provided (and no mask), runs image-to-image using this as the starting point. "
            "PNG/JPEG recommended. Combine with prompt to guide the transformation."
        ),
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    mask: Optional[str] = Field(
        default=None,
        description=(
            "Base64 string image mask for inpainting. Must be provided together with 'image'. "
            "Non-zero/opaque regions indicate areas to modify. Triggers inpainting when supported."
        ),
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    references: list[References] = Field(
        default_factory=list,
        description="Optional control/adapters (style/depth/canny/pose/etc.) applied across models when supported.",
    )

    @property
    def inferred_mode(self) -> InferredMode:
        if self.mask and self.image:
            return "inpainting"
        if self.image:
            return "image-to-image"
        return "text-to-image"

    @property
    def meta(self) -> ImagesModelInfo:
        return MODEL_META[self.model]

    @property
    def external_model(self) -> bool:
        return self.meta.external

    @property
    def task_name(self) -> str:
        return f"images.{self.model}"

    @property
    def task_queue(self) -> str:
        return self.meta.queue

    @property
    def cleaned_prompt(self) -> str:
        return " ".join(self.prompt.split())

    @model_validator(mode="after")
    def _validate_capabilities(self):
        mode = self.inferred_mode
        if not self.meta.supports_inferred_mode(mode):
            raise ValueError(f"Model '{self.model}' does not support mode '{mode}'.")
        if self.mask and not self.image:
            raise ValueError("mask requires image.")
        if self.references and not self.meta.references:
            raise ValueError(f"Model '{self.model}' does not support references.")
        return self


class ImageWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class ImageResponse(BaseModel):
    id: UUID
    status: TaskStatus
    result: Optional[ImageWorkerResponse] = None
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


class ImageCreateResponse(BaseModel):
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


class ImageModelsResponse(BaseModel):
    models: Dict[ModelName, ImagesModelInfo]

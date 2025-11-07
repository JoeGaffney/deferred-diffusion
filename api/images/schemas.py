from typing import Any, Dict, Literal, Optional, TypeAlias, Union, get_args
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field, model_validator

# User facing choice
ModelName: TypeAlias = Literal[
    "sd-xl",
    "sd-3",
    "flux-1",
    "qwen-image",
    "depth-anything-2",
    "segment-anything-2",
    "real-esrgan-x4",
    "gpt-image-1",
    "runway-gen4-image",
    "flux-1-pro",
    "topazlabs-upscale",
    "google-gemini-2",
    "bytedance-seedream-4",
]
InferredMode: TypeAlias = Literal["text_to_image", "image_to_image", "inpainting"]
Provider: TypeAlias = Literal["local", "openai", "replicate", "runway"]


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
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        references=True,
        description="Stable Diffusion XL variant with broad adapter/control support.",
    ),
    "sd-3": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        description="Stable Diffusion 3.5 for complex compositions.",
    ),
    "flux-1": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        references=True,
        description="FLUX dev model (Krea tuned). Uses Kontext for img2img, Fill for inpainting.",
    ),
    "qwen-image": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        references=True,
        description="Qwen image generation and manipulation.",
    ),
    "depth-anything-2": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"image_to_image"},
        description="Depth estimation pipeline.",
    ),
    "segment-anything-2": ImagesModelInfo(
        provider="local",
        external=False,
        supported_modes={"image_to_image"},
        description="Segmentation pipeline.",
    ),
    "gpt-image-1": ImagesModelInfo(
        provider="openai",
        external=True,
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        references=True,
        description="OpenAI image model.",
    ),
    "runway-gen4-image": ImagesModelInfo(
        provider="runway",
        external=True,
        supported_modes={"text_to_image", "image_to_image"},
        references=True,
        description="Runway Gen-4 image model.",
    ),
    "flux-1-pro": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_image", "image_to_image", "inpainting"},
        description="FLUX 1.1 Pro variants via external provider.",
    ),
    "topazlabs-upscale": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"image_to_image"},
        description="Topaz upscale model.",
    ),
    "google-gemini-2": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_image", "image_to_image"},
        references=True,
        description="Gemini multimodal image model.",
    ),
    "bytedance-seedream-4": ImagesModelInfo(
        provider="replicate",
        external=True,
        supported_modes={"text_to_image", "image_to_image"},
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
    mode: Literal["style", "style-plus", "face", "depth", "canny", "pose"]
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
    high_quality: bool = Field(
        default=False,
        description="Use higher quality models and steps when available. May increase cost/latency. For example some external models have pro variants.",
    )

    @property
    def inferred_mode(self) -> InferredMode:
        if self.mask and self.image:
            return "inpainting"
        if self.image:
            return "image_to_image"
        return "text_to_image"

    @property
    def meta(self) -> ImagesModelInfo:
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
        if self.mask and not self.image:
            raise ValueError("mask requires image.")
        if self.references and not self.meta.references:
            raise ValueError(f"Model '{self.model}' does not support references.")
        return self


class ImageWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class ImageResponse(BaseModel):
    id: UUID
    status: str
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
    status: str
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

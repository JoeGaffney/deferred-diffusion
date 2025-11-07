from typing import Any, Dict, Literal, Optional, TypeAlias, Union, get_args
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field, model_validator

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
InferredMode: TypeAlias = Literal["text_to_image", "image_to_image", "image_to_image_inpainting"]


class ModelInfo(BaseModel):
    provider: str = Field(description="Source/provider identifier (local, openai, runway, replicate, etc.)")
    external: bool = Field(description="True if the model is invoked via an external API")
    text: bool = False
    image: bool = False
    mask: bool = False
    references: bool = False
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"

    def supports_inferred_mode(self, mode: InferredMode) -> bool:
        if mode == "text_to_image":
            return self.text
        if mode == "image_to_image":
            return self.image
        if mode == "image_to_image_inpainting":
            return self.image and self.mask
        return False


# Unified metadata
MODEL_META: Dict[ModelName, ModelInfo] = {
    "sd-xl": ModelInfo(
        provider="local",
        external=False,
        text=True,
        image=True,
        mask=True,
        references=True,
        description="Stable Diffusion XL variant with broad adapter/control support.",
    ),
    "sd-3": ModelInfo(
        provider="local",
        external=False,
        text=True,
        image=True,
        mask=True,
        description="Stable Diffusion 3.5 for complex compositions.",
    ),
    "flux-1": ModelInfo(
        provider="local",
        external=False,
        text=True,
        image=True,
        mask=True,
        references=True,
        description="FLUX dev model (Krea tuned). Uses Kontext for img2img, Fill for inpainting.",
    ),
    "qwen-image": ModelInfo(
        provider="local",
        external=False,
        text=True,
        image=True,
        mask=True,
        references=True,
        description="Qwen image generation and manipulation.",
    ),
    "depth-anything-2": ModelInfo(
        provider="local",
        external=False,
        image=True,
        description="Depth estimation pipeline.",
    ),
    "segment-anything-2": ModelInfo(
        provider="local",
        external=False,
        image=True,
        description="Segmentation pipeline.",
    ),
    "gpt-image-1": ModelInfo(
        provider="openai",
        external=True,
        text=True,
        image=True,
        mask=True,
        references=True,
        description="OpenAI image model.",
    ),
    "runway-gen4-image": ModelInfo(
        provider="runway",
        external=True,
        text=True,
        image=True,
        references=True,
        description="Runway Gen-4 image model.",
    ),
    "flux-1-pro": ModelInfo(
        provider="replicate",
        external=True,
        text=True,
        image=True,
        mask=True,
        description="FLUX 1.1 Pro variants via external provider.",
    ),
    "topazlabs-upscale": ModelInfo(
        provider="replicate",
        external=True,
        image=True,
        description="Topaz upscale model.",
    ),
    "google-gemini-2": ModelInfo(
        provider="replicate",
        external=True,
        text=True,
        image=True,
        references=True,
        description="Gemini multimodal image model.",
    ),
    "bytedance-seedream-4": ModelInfo(
        provider="replicate",
        external=True,
        text=True,
        image=True,
        references=True,
        description="Seedream image-to-image context model.",
    ),
}


def generate_model_docs():
    header = (
        "# Image Models\n"
        "External models proxy to provider APIs; local models run on your GPU.\n\n"
        "| Model | Provider | External | Queue | Text | Image | Mask | References | Description |\n"
        "|-------|----------|:--------:|:-----:|:----:|:-----:|:----:|:----------:|-------------|\n"
    )
    rows = []
    for name, meta in MODEL_META.items():
        rows.append(
            f"| {name} | {meta.provider} | {'Yes' if meta.external else 'No'} | {meta.queue} | "
            f"{'✓' if meta.text else '✗'} | {'✓' if meta.image else '✗'} | {'✓' if meta.mask else '✗'} | "
            f"{'✓' if meta.references else '✗'} | {meta.description or ''} |"
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
            return "image_to_image_inpainting"
        if self.image:
            return "image_to_image"
        return "text_to_image"

    @property
    def meta(self) -> ModelInfo:
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


class ImageCreateResponse(BaseModel):
    id: UUID
    status: str

    class Config:
        json_schema_extra = {
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "PENDING",
            }
        }

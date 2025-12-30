from typing import Dict, List, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.schemas import Base64Image, Base64ResponseBytes, Provider, TaskStatus

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
    "runway-gen-4",
    "flux-1-pro",
    "flux-2-pro",
    "topazlabs-upscale",
    "gemini-2",
    "gemini-3",
    "seedream-4",
]


class ImagesModelInfo(BaseModel):
    external: bool = Field(description="True if the model is invoked via an external API")
    provider: Provider = Field(description="Source/provider identifier")
    text_to_image: bool = False
    image_to_image: bool = False
    inpainting: bool = False
    references: bool = False
    description: Optional[str] = None

    @property
    def queue(self) -> str:
        return "cpu" if self.external else "gpu"


# Unified metadata
MODEL_META: Dict[ModelName, ImagesModelInfo] = {
    "sd-xl": ImagesModelInfo(
        provider="local",
        external=False,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        references=False,
        description="Stable Diffusion XL variant.",
    ),
    "flux-1": ImagesModelInfo(
        provider="local",
        external=False,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        references=False,
        description="FLUX dev model (Krea tuned). Uses Kontext for image-to-image, Fill for inpainting.",
    ),
    "flux-2": ImagesModelInfo(
        provider="local",
        external=False,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        references=True,
        description="FLUX 2.0 dev model with edit capabilities.",
    ),
    "qwen-image": ImagesModelInfo(
        provider="local",
        external=False,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        references=True,
        description="Qwen image generation and manipulation.",
    ),
    "z-image": ImagesModelInfo(
        provider="local",
        external=False,
        text_to_image=True,
        description="Z-Image open-source image generation model.",
    ),
    "depth-anything-2": ImagesModelInfo(
        provider="local",
        external=False,
        image_to_image=True,
        description="Depth estimation pipeline.",
    ),
    "sam-2": ImagesModelInfo(
        provider="local",
        external=False,
        image_to_image=True,
        description="Meta's SAM 2 Segmentation pipeline.",
    ),
    "sam-3": ImagesModelInfo(
        provider="local",
        external=False,
        image_to_image=True,
        description="Meta's SAM 3 Segmentation pipeline. (Broken atm)",
    ),
    "gpt-image-1": ImagesModelInfo(
        provider="openai",
        external=True,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        references=True,
        description="GPT Image 1.5 is OpenAI's latest image generation model, built for production-quality visuals and controllable creative workflows.",
    ),
    "runway-gen-4": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        references=True,
        description="Runway Gen-4 image model.",
    ),
    "flux-1-pro": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        description="FLUX 1.1 Pro variants via external provider.",
    ),
    "flux-2-pro": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        inpainting=True,
        description="FLUX 2.0 Pro variants via external provider.",
        references=True,
    ),
    "topazlabs-upscale": ImagesModelInfo(
        provider="replicate",
        external=True,
        image_to_image=True,
        description="Topaz upscale model.",
    ),
    "gemini-2": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        references=True,
        description="Googles Gemini 2.5 multimodal image model (aka 'Nano Banana').",
    ),
    "gemini-3": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        references=True,
        description="Googles Gemini 3 Pro multimodal image model (aka 'Nano Banana Pro').",
    ),
    "seedream-4": ImagesModelInfo(
        provider="replicate",
        external=True,
        text_to_image=True,
        image_to_image=True,
        references=True,
        description="Bytedances Seedream 4.5: Upgraded Bytedance image model with stronger spatial understanding and world knowledge.",
    ),
}


def generate_model_docs():
    header = "# Image Models\n" "External models proxy to provider APIs; local models run on your GPU.\n\n"

    # Get all fields from ImagesModelInfo to build table header
    model_fields = ImagesModelInfo.model_fields
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


class References(BaseModel):
    image: Base64Image = Field(
        description="Base64 image string",
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
    image: Optional[Base64Image] = Field(
        default=None,
        description=(
            "Base64 string image. If provided (and no mask), runs image-to-image using this as the starting point. "
            "PNG/JPEG recommended. Combine with prompt to guide the transformation."
        ),
    )
    mask: Optional[Base64Image] = Field(
        default=None,
        description=(
            "Base64 string image mask for inpainting. Must be provided together with 'image'. "
            "Non-zero/opaque regions indicate areas to modify. Triggers inpainting when supported."
        ),
    )
    references: list[References] = Field(
        default_factory=list,
        description="Optional reference images that modern models can use to guide image generation.",
    )

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
        if self.mask and self.image:
            # Inpainting mode
            if not self.meta.inpainting:
                raise ValueError(f"Model '{self.model}' does not support inpainting.")
        elif self.image:
            # Image-to-image mode
            if not self.meta.image_to_image:
                raise ValueError(f"Model '{self.model}' does not support image_to_image.")
        else:
            # Text-to-image mode
            if not self.meta.text_to_image:
                raise ValueError(f"Model '{self.model}' does not support text_to_image.")

        if self.mask and not self.image:
            raise ValueError("mask requires image.")
        if self.references and not self.meta.references:
            raise ValueError(f"Model '{self.model}' does not support references.")
        return self


class ImageWorkerResponse(BaseModel):
    logs: List[str] = []
    base64_data: Optional[Base64ResponseBytes] = None
    url: Optional[str] = Field(None, description="Signed URL to download the image")


class ImageResponse(BaseModel):
    id: UUID
    status: TaskStatus
    result: Optional[ImageWorkerResponse] = None
    error_message: Optional[str] = None
    logs: List[str] = []
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "9a34ab0a-9e9a-4b84-90f7-d8b30c59b6ae",
                "status": "SUCCESS",
                "result": {
                    "base64_data": "iVBORw0KGgoAAAANSUhEUgAA...",
                },
                "error_message": None,
                "logs": ["Setup", "Progress: 10%", "Progress: 20%", "..."],
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

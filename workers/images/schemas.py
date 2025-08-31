from typing import Any, Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

# User facing choice
ModelName: TypeAlias = Literal[
    "sd-xl",
    "sd-3",
    "flux-1",
    "flux-1-krea",
    "flux-kontext-1",
    "qwen-image",
    "depth-anything-2",
    "segment-anything-2",
    "real-esrgan-x4",
    "gpt-image-1",
    "runway-gen4-image",
    "flux-kontext-1-pro",
    "flux-1-1-pro",
    "topazlabs-upscale",
]

# External-only model names (convenience alias)
ModelNameExternal: TypeAlias = Literal[
    "gpt-image-1",
    "runway-gen4-image",
    "flux-kontext-1-pro",
    "flux-1-1-pro",
    "topazlabs-upscale",
]

# Task names used for routing
TaskName: TypeAlias = Literal["process_image", "process_image_external"]

# No ModelInfo or ModelFamily here: workers control model paths/weights and implementation.
# Lightweight MODEL_META (below) is the single source of truth for docs/routing.

# Lightweight metadata for documentation and routing only.
# Workers are responsible for model paths, weights and other infra details.
MODEL_META: Dict[ModelName, Dict[str, Any]] = {
    "sd-xl": {
        "external": False,
        "family": "sdxl",
        "path": "SG161222/RealVisXL_V4.0",
        "inpainting_path": "OzzyGT/RealVisXL_V4.0_inpainting",
        "references": True,
        "description": "Stable Diffusion XL variant that supports adapters and inpainting. High-quality, detailed images for complex prompts.",
    },
    "sd-3": {
        "external": False,
        "family": "sd3",
        "path": "stabilityai/stable-diffusion-3.5-large",
        "description": "Stable Diffusion 3.5 with improved prompt understanding and composition.",
    },
    "flux-1": {
        "external": False,
        "family": "flux",
        "path": "black-forest-labs/FLUX.1-dev",
        "inpainting_path": "black-forest-labs/FLUX.1-Fill-dev",
        "references": True,
        "description": "FLUX.1 specializes in text rendering and coherent scene composition, great for illustrations and UI mockups.",
    },
    "flux-1-krea": {
        "external": False,
        "family": "flux",
        "path": "black-forest-labs/FLUX.1-Krea-dev",
        "references": True,
        "description": "Krea-opinion variant of FLUX.1 that biases toward photorealism.",
    },
    "flux-kontext-1": {
        "external": False,
        "family": "flux_kontext",
        "path": "black-forest-labs/FLUX.1-Kontext-dev",
        "description": "Context-aware FLUX model for scene continuation and thematically consistent generations.",
    },
    "qwen-image": {
        "external": False,
        "family": "qwen",
        "path": "ovedrive/qwen-image-4bit",
        "image_to_image_path": "ovedrive/qwen-image-edit-4bit",
        "description": "Qwen image model optimized for nuanced prompt understanding and detailed outputs.",
    },
    "depth-anything-2": {
        "external": False,
        "family": "depth_anything",
        "path": "depth-anything/Depth-Anything-V2-Large-hf",
        "description": "Depth estimation model producing high-quality depth maps for 3D and ControlNet use.",
    },
    "segment-anything-2": {
        "external": False,
        "family": "segment_anything",
        "path": "sam2.1_hiera_base_plus",
        "description": "State-of-the-art segmentation model for object and feature extraction.",
    },
    "real-esrgan-x4": {
        "external": False,
        "family": "real_esrgan",
        "path": "weights/RealESRGAN_x4plus.pth",
        "description": "Real-ESRGAN x4 upscaler for improving image resolution while preserving detail.",
    },
    "gpt-image-1": {
        "external": True,
        "family": "openai",
        "path": "gpt-image-1",
        "references": True,
        "description": "OpenAI's image API for high-fidelity, instruction-following image generation.",
    },
    "runway-gen4-image": {
        "external": True,
        "family": "runway",
        "path": "gen4_image",
        "references": True,
        "description": "Runway Gen-4 image model delivering coherent, high-quality images via API.",
    },
    "flux-kontext-1-pro": {
        "external": True,
        "family": "flux_kontext",
        "path": "black-forest-labs/flux-kontext-pro",
        "description": "Pro (API) version of FLUX Kontext with enhanced contextual capabilities.",
    },
    "flux-1-1-pro": {
        "external": True,
        "family": "flux",
        "path": "black-forest-labs/flux-1.1-pro",
        "inpainting_path": "black-forest-labs/flux-fill-pro",
        "description": "Flux 1.1 pro via API: improved text rendering and professional quality outputs.",
    },
    "topazlabs-upscale": {
        "external": True,
        "family": "topazlabs",
        "path": "topazlabs/image-upscale",
        "description": "Topaz Labs API-based upscaler focused on photographic detail preservation.",
    },
}

# Derive external models from MODEL_META to keep a single source of truth
EXTERNAL_MODELS = {name for name, meta in MODEL_META.items() if meta.get("external")}


def generate_model_docs() -> str:
    """Generate a brief documentation string listing available models and metadata.

    Shows family/path/inpainting/image-to-image when available; still informational
    only — the workers are the source of truth for actual implementation.
    """
    docs = "# Available Models\n\nThis document lists the available model names and a short, informational description. Paths and families are informational only; workers control the real implementation.\n\n"
    for model_name, meta in MODEL_META.items():
        docs += f"## {model_name}\n\n{meta.get('description','')}\n\n"
        if meta.get("path"):
            docs += f"- **Path hint:** `{meta.get('path')}`\n"
        if meta.get("inpainting_path"):
            docs += f"- **Inpainting path hint:** `{meta.get('inpainting_path')}`\n"
        if meta.get("image_to_image_path"):
            docs += f"- **Image-to-image path hint:** `{meta.get('image_to_image_path')}`\n"
        if meta.get("family"):
            docs += f"- **Family:** {meta.get('family')}\n"
        docs += f"- **External API:** {'Yes' if meta.get('external') else 'No'}\n"
        docs += f"- **References:** {'✓' if meta.get('references') else '✗'}\n\n"
    return docs


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
    negative_prompt: str = Field(
        default="worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes",
        description="Negative prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    height: int = 512
    width: int = 512
    num_inference_steps: int = 25
    seed: int = 42
    guidance_scale: float = 5.0
    strength: float = 0.5
    image: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
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
    references: list[References] = []

    @property
    def external_model(self) -> bool:
        """Determine if a model is external purely by model name."""
        return self.model in EXTERNAL_MODELS

    @property
    def task_name(self) -> ModelName:
        return self.model

    @property
    def task_queue(self) -> str:
        """Return the task queue based on whether the model is external or not."""
        return "cpu" if self.external_model else "gpu"


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

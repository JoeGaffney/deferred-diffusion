from typing import Any, Dict, Literal, Optional, TypeAlias, Union, get_args
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelNameLocal: TypeAlias = Literal[
    "sd-xl",
    "sd-3",
    "flux-1",
    "qwen-image",
    "depth-anything-2",
    "segment-anything-2",
    "real-esrgan-x4",
]


# External-only model names (convenience alias)
ModelNameExternal: TypeAlias = Literal[
    "gpt-image-1",
    "runway-gen4-image",
    "flux-1-pro",
    "topazlabs-upscale",
    "google-gemini-2",
    "bytedance-seedream-4",
]


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


class ModelInfo(BaseModel):
    references: bool = Field(default=False, description="Supports image references as input")
    image: bool = Field(default=False, description="Supports input image for image-to-image generation")
    mask: bool = Field(default=False, description="Supports input mask for inpainting")
    text: bool = Field(default=False, description="Supports text-to-image generation")
    description: Optional[str] = None

    def to_doc_format(self, model_name: str) -> str:
        """Generate documentation for this model"""
        doc = f"## {model_name}\n\n"
        doc += f"{self.description}\n"
        doc += f"- **Text (text-to-image):** {'✓' if self.text else '✗'}\n"
        doc += f"- **Image (image-to-image):** {'✓' if self.image else '✗'}\n"
        doc += f"- **Mask (inpainting):** {'✓' if self.mask else '✗'}\n"
        doc += f"- **References (controlnets and adapters):** {'✓' if self.references else '✗'}\n"

        doc += "\n"
        return doc


MODEL_META_LOCAL: Dict[ModelNameLocal, ModelInfo] = {
    "sd-xl": ModelInfo(
        references=True,
        text=True,
        image=True,
        mask=True,
        description="Stable Diffusion XL variant supports the most control nets and IP adapters. It excels at generating high-quality, detailed images with complex prompts and multiple subjects.",
    ),
    "sd-3": ModelInfo(
        text=True,
        image=True,
        mask=True,
        description="Stable Diffusion 3.5 offers superior prompt understanding and composition. Excels at complex scenes, concept art, and handling multiple subjects with accurate interactions.",
    ),
    "flux-1": ModelInfo(
        text=True,
        image=True,
        mask=True,
        references=True,
        description="FLUX Krea model is flux-1 dev trained with opinions from krea for more photorealistic results. Will use flux Kontext for image to image and flux fill for inpainting.",
    ),
    "qwen-image": ModelInfo(
        text=True,
        image=True,
        mask=True,
        references=True,
        description="Qwen model specializes in generating high-quality images from textual descriptions. It excels at understanding nuanced prompts and delivering detailed visuals.",
    ),
    "depth-anything-2": ModelInfo(
        image=True,
        description="Advanced depth estimation model. Creates high-quality depth maps from any image for 3D visualization, AR applications, and as input for ControlNet pipelines.",
    ),
    "segment-anything-2": ModelInfo(
        image=True,
        description="State-of-the-art image segmentation model. Precisely identifies and segments objects, people, and features for compositing, editing, and analysis.",
    ),
}

MODEL_META_EXTERNAL: Dict[ModelNameExternal, ModelInfo] = {
    "gpt-image-1": ModelInfo(
        references=True,
        text=True,
        image=True,
        mask=True,
        description="OpenAI's advanced image generation model with exceptional understanding of complex prompts. Excels at photorealistic imagery, accurate object rendering, and following detailed instructions.",
    ),
    "runway-gen4-image": ModelInfo(
        references=True,
        text=True,
        image=True,
        description="Runway's Gen-4 image model delivering high-fidelity results with strong coherence. Particularly good at combinging multiple references into a single, cohesive image.",
    ),
    "flux-1-pro": ModelInfo(
        text=True,
        image=True,
        mask=True,
        description="Pro variants of FLUX 1.1 with enhanced capabilities. Will use flux Kontext pro for image to image and flux fill pro for inpainting.",
    ),
    "topazlabs-upscale": ModelInfo(
        image=True,
        description="Topaz Labs' advanced image upscaling model. Specializes in enhancing image resolution while preserving fine details and textures, ideal for professional photography and print work.",
    ),
    "google-gemini-2": ModelInfo(
        text=True,
        image=True,
        references=True,
        description="Google's Gemini 2.5 model for advanced image generation and manipulation. Aka nano bannana",
    ),
    "bytedance-seedream-4": ModelInfo(
        text=True,
        image=True,
        references=True,
        description="Supreme image to image context model from Bytedance. Excels at transforming input images based on textual prompts while maintaining core elements of the original image.",
    ),
}


def generate_model_docs(local=True):
    docs = """ # Generate images using various diffusion models.
- External models are processed through their respective APIs.
- Local models are processed on your own GPU workers.
- ControlNets and Adapters are unified as **references**.
"""
    if local:
        for model_name, model_info in MODEL_META_LOCAL.items():
            docs += model_info.to_doc_format(model_name)
    else:
        for model_name, model_info in MODEL_META_EXTERNAL.items():
            docs += model_info.to_doc_format(model_name)
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
    prompt: str = Field(
        default="Detailed, 8k, photorealistic",
        description="Positive Prompt text",
        json_schema_extra={"format": "multi_line"},
    )
    height: int = 720
    width: int = 1280
    seed: int = 42
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
    high_quality: bool = Field(
        default=False,
        description="Use high quality model variant when available (may cost more and take longer). Will use higher steps in local models.",
    )


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

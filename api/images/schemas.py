from typing import Any, Dict, Literal, Optional, Tuple
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field


class ControlNetSchema(BaseModel):
    model: Literal[
        "pose",
        "depth",
        "canny",
    ]
    conditioning_scale: float = 0.5
    image: str = Field(
        description="Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/png, image/jpg, image/jpeg",  # Or "image/*" if you accept multiple formats
        },
    )


class IpAdapterModelConfig(BaseModel):
    model: str = Field(
        description="The model name for the IP adapter.",
    )
    subfolder: str = Field(
        description="The subfolder where the IP adapter model is stored.",
    )
    weight_name: str = Field(
        description="The weight name for the IP adapter model.",
    )
    image_encoder: bool = Field(description="Whether to use the image encoder for the IP adapter model.")
    image_encoder_subfolder: str = Field(description="The subfolder where the image encoder model is stored.")


class IpAdapterModel(BaseModel):
    model: Literal[
        "style",
        "style-plus",
        "face",
    ]
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
    scale: float = 0.5
    scale_layers: str = "all"


class ImageRequest(BaseModel):
    model: Literal[
        "sd1.5",
        "sdxl",
        "sdxl-refiner",
        "RealVisXL",
        "Fluently-XL",
        "juggernaut-xl",
        "sd3",
        "sd3.5",
        "flux-schnell",
        "flux-dev",
        "depth-anything",
        "segment-anything",
        "sd-x4-upscaler",
        "gpt-image-1",
        "runway/gen4_image",
        "HiDream",
    ]
    comfy_workflow: Optional[Dict[str, Any]] = None
    prompt: str = "Detailed, 8k, photorealistic"
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    guidance_scale: float = 5.0
    max_height: int = 2048
    max_width: int = 2048
    num_inference_steps: int = 25
    seed: int = 42
    strength: float = 0.5
    inpainting_full_image: bool = True
    target_precision: Literal[4, 8, 16] = Field(
        8, description="Global target precision for quantization; applied selectively per model and component."
    )
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
    ip_adapters: list[IpAdapterModel] = []
    controlnets: list[ControlNetSchema] = []


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


class ModelConfig(BaseModel):
    model_path: str
    model_family: str
    mode: str = "auto"

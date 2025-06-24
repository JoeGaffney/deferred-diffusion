from typing import Any, Dict, Literal, Optional, TypeAlias
from uuid import UUID

from pydantic import Base64Bytes, BaseModel, Field

ModelName: TypeAlias = Literal[
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
    "HiDream",
    "gpt-image-1",
    "runway/gen4_image",
    "flux-kontext-pro",
    "flux-pro",
]

ModelFamily: TypeAlias = Literal[
    "sd1.5",
    "sdxl",
    "sd3",
    "hidream",
    "flux",
    "openai",
    "runway",
    "sd_upscaler",
    "segment_anything",
    "depth_anything",
    "flux_kontext",
    "flux_pro",
]


class ModelInfo(BaseModel):
    family: ModelFamily
    path: str
    external: bool


MODEL_CONFIG: Dict[ModelName, ModelInfo] = {
    "sd1.5": ModelInfo(family="sd1.5", path="stable-diffusion-v1-5/stable-diffusion-v1-5", external=False),
    "sdxl": ModelInfo(family="sdxl", path="stabilityai/stable-diffusion-xl-base-1.0", external=False),
    "sdxl-refiner": ModelInfo(family="sdxl", path="stabilityai/stable-diffusion-xl-refiner-1.0", external=False),
    "RealVisXL": ModelInfo(family="sdxl", path="SG161222/RealVisXL_V4.0", external=False),
    "Fluently-XL": ModelInfo(family="sdxl", path="fluently/Fluently-XL-v4", external=False),
    "juggernaut-xl": ModelInfo(family="sdxl", path="RunDiffusion/Juggernaut-XL-v9", external=False),
    "sd3": ModelInfo(family="sd3", path="stabilityai/stable-diffusion-3-medium-diffusers", external=False),
    "sd3.5": ModelInfo(family="sd3", path="stabilityai/stable-diffusion-3.5-medium", external=False),
    "flux-schnell": ModelInfo(family="flux", path="black-forest-labs/FLUX.1-schnell", external=False),
    "flux-dev": ModelInfo(family="flux", path="black-forest-labs/FLUX.1-dev", external=False),
    "depth-anything": ModelInfo(
        family="depth_anything", path="depth-anything/Depth-Anything-V2-Large-hf", external=False
    ),
    "segment-anything": ModelInfo(family="segment_anything", path="sam2.1_hiera_base_plus", external=False),
    "sd-x4-upscaler": ModelInfo(family="sd_upscaler", path="stabilityai/stable-diffusion-x4-upscaler", external=False),
    "HiDream": ModelInfo(family="hidream", path="HiDream-ai/HiDream-I1-Full", external=False),
    "gpt-image-1": ModelInfo(family="openai", path="gpt-image-1", external=True),
    "runway/gen4_image": ModelInfo(family="runway", path="gen4_image", external=True),
    "flux-kontext-pro": ModelInfo(family="flux_kontext", path="black-forest-labs/flux-kontext-pro", external=True),
    "flux-pro": ModelInfo(family="flux_pro", path="black-forest-labs/flux-1.1-pro", external=True),
}

TaskName: TypeAlias = Literal["process_image", "process_image_external"]


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


class IpAdapterModel(BaseModel):
    model: Literal[
        "style",
        "style-plus",
        "face",
    ]
    scale: float = 0.5
    scale_layers: str = "all"
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
        default="worst quality, inconsistent motion, blurry, jittery, distorted",
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
    ip_adapters: list[IpAdapterModel] = []
    controlnets: list[ControlNetSchema] = []

    @property
    def model_family(self) -> ModelFamily:
        return MODEL_CONFIG[self.model].family

    @property
    def model_path(self) -> str:
        return MODEL_CONFIG[self.model].path

    @property
    def external_model(self) -> bool:
        return MODEL_CONFIG[self.model].external

    @property
    def task_name(self) -> TaskName:
        return "process_image_external" if self.external_model else "process_image"


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

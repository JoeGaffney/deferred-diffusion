from typing import Any, Dict, Literal, Optional, Tuple, TypeAlias

from pydantic import Base64Bytes, BaseModel, Field
from torch import dtype

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
    "gpt-image-1",
    "runway/gen4_image",
    "HiDream",
]

ModelFamily: TypeAlias = Literal[
    "sd1.5", "sdxl", "sd3", "hidream", "flux", "openai", "runway", "sd_upscaler", "segment_anything", "depth_anything"
]


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


class ComfyWorkflow(BaseModel):
    """Represents a ComfyUI workflow with dynamic node structure."""

    model_config = {
        "extra": "allow",
    }


class ImageRequest(BaseModel):
    model: ModelName
    comfy_workflow: Optional[ComfyWorkflow] = Field(
        default=None,
        description="ComfyUI workflow configuration with dynamic node structure",
    )
    controlnets: list[ControlNetSchema] = []
    guidance_scale: float = 5.0
    image: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    ip_adapters: list[IpAdapterModel] = []
    mask: Optional[str] = Field(
        default=None,
        description="Optional Base64 image string",
        json_schema_extra={
            "contentEncoding": "base64",
            "contentMediaType": "image/*",
        },
    )
    max_height: int = 2048
    max_width: int = 2048
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_inference_steps: int = 25
    target_precision: Literal[4, 8, 16] = Field(
        description="Global target precision for quantization; applied selectively per model and component.",
        default=8,
    )
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5

    @property
    def model_family(self) -> ModelFamily:
        mapping: Dict[ModelName, ModelFamily] = {
            "sd1.5": "sd1.5",
            "sdxl": "sdxl",
            "sdxl-refiner": "sdxl",
            "RealVisXL": "sdxl",
            "Fluently-XL": "sdxl",
            "juggernaut-xl": "sdxl",
            "sd3": "sd3",
            "sd3.5": "sd3",
            "flux-schnell": "flux",
            "flux-dev": "flux",
            "depth-anything": "depth_anything",
            "segment-anything": "segment_anything",
            "sd-x4-upscaler": "sd_upscaler",
            "gpt-image-1": "openai",
            "runway/gen4_image": "runway",
            "HiDream": "hidream",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model family defined for model '{self.model}'")

    @property
    def model_path(self) -> str:
        mapping: Dict[ModelName, str] = {
            "sd1.5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
            "sdxl-refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
            "RealVisXL": "SG161222/RealVisXL_V4.0",
            "Fluently-XL": "fluently/Fluently-XL-v4",
            "juggernaut-xl": "RunDiffusion/Juggernaut-XL-v9",
            "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
            "sd3.5": "stabilityai/stable-diffusion-3.5-medium",
            "flux-schnell": "black-forest-labs/FLUX.1-schnell",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "depth-anything": "depth-anything/Depth-Anything-V2-Large-hf",
            "segment-anything": "sam2.1_hiera_base_plus",
            "sd-x4-upscaler": "stabilityai/stable-diffusion-x4-upscaler",
            "gpt-image-1": "gpt-image-1",
            "runway/gen4_image": "gen4_image",
            "HiDream": "HiDream-ai/HiDream-I1-Full",
        }
        try:
            return mapping[self.model]
        except KeyError:
            raise ValueError(f"No model path defined for model '{self.model}'")

    @property
    def external_model(self) -> bool:
        external_models = [
            "openai",
            "runway",
        ]
        return self.model_family in external_models


class ImageWorkerResponse(BaseModel):
    base64_data: Base64Bytes


class PipelineConfig(BaseModel):
    model_id: str
    model_family: ModelFamily
    torch_dtype: dtype
    target_precision: Literal[4, 8, 16] = Field(
        8, description="Global target precision for quantization; applied selectively per model and component."
    )
    use_safetensors: bool
    ip_adapter_models: Tuple[str, ...]
    ip_adapter_subfolders: Tuple[str, ...]
    ip_adapter_weights: Tuple[str, ...]
    ip_adapter_image_encoder_model: str
    ip_adapter_image_encoder_subfolder: str

    class Config:
        frozen = True  # Makes the model immutable/hashable
        arbitrary_types_allowed = True  # Needed for torch.dtype

    def __hash__(self):
        return hash(
            (
                self.model_id,
                self.torch_dtype,
                self.target_precision,
                self.use_safetensors,
                self.ip_adapter_models,
                self.ip_adapter_subfolders,
                self.ip_adapter_weights,
                self.ip_adapter_image_encoder_model,
                self.ip_adapter_image_encoder_subfolder,
            )
        )

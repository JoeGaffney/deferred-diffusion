from typing import Literal, Tuple

from pydantic import BaseModel, Field
from torch import dtype


class ControlNetSchema(BaseModel):
    model: Literal[
        "pose",
        "depth",
        "canny",
    ]
    conditioning_scale: float = 0.5
    image_path: str


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
    image_path: str
    mask_path: str = ""
    scale: float = 0.5
    scale_layers: str = "all"


class ImageRequest(BaseModel):
    model: Literal[
        "sd1.5",
        "sdxl",
        "sdxl-refiner",
        "RealVisXL",
        "Fluently-XL",
        "sd3",
        "sd3.5",
        "flux-schnell",
        "depth-anything",
        "segment-anything",
        "sd-x4-upscaler",
    ]
    controlnets: list[ControlNetSchema] = []
    optimize_low_vram: bool = False
    guidance_scale: float = 5.0
    inpainting_full_image: bool = True
    input_image_path: str = ""
    input_mask_path: str = ""
    ip_adapters: list[IpAdapterModel] = []
    max_height: int = 2048
    max_width: int = 2048
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_inference_steps: int = 25
    output_image_path: str = ""
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5


class ImageResponse(BaseModel):
    data: str


class ModelConfig(BaseModel):
    model_path: str
    model_family: str
    guf_path: str
    mode: str = "auto"


class PipelineConfig(BaseModel):
    model_id: str
    model_family: str
    model_guf_path: str
    torch_dtype: dtype
    optimize_low_vram: bool
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
                self.optimize_low_vram,
                self.use_safetensors,
                self.ip_adapter_models,
                self.ip_adapter_subfolders,
                self.ip_adapter_weights,
                self.ip_adapter_image_encoder_model,
                self.ip_adapter_image_encoder_subfolder,
            )
        )

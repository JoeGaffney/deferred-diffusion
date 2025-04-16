from typing import Tuple

from pydantic import BaseModel, Field
from torch import dtype


class ControlNetSchema(BaseModel):
    conditioning_scale: float = 0.5
    image_path: str
    model: str


class IpAdapterModel(BaseModel):
    image_path: str
    mask_path: str = ""
    model: str = Field("h94/IP-Adapter", min_length=1)
    scale: float = 0.5
    scale_layers: str = "all"
    subfolder: str = Field("models", min_length=1)
    weight_name: str = Field("ip-adapter_sd15.bin", min_length=1)
    image_encoder: bool = False


class ImageRequest(BaseModel):
    controlnets: list[ControlNetSchema] = []
    optimize_low_vram: bool = True
    guidance_scale: float = 5.0
    inpainting_full_image: bool = True
    input_image_path: str = ""
    input_mask_path: str = ""
    ip_adapters: list[IpAdapterModel] = []
    max_height: int = 2048
    max_width: int = 2048
    model: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_inference_steps: int = 25
    output_image_path: str = ""
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5


class ImageResponse(BaseModel):
    data: str


class PipelineConfig(BaseModel):
    model_id: str
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

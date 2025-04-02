from pydantic import BaseModel, Field


class ImageResponse(BaseModel):
    data: str


class ControlNet(BaseModel):
    conditioning_scale: float = 0.5
    image_path: str
    model: str


class IpAdapterModel(BaseModel):
    image_path: str
    model: str = Field("h94/IP-Adapter", min_length=1)
    scale: float = 0.5
    subfolder: str = Field("models", min_length=1)
    weight_name: str = Field("ip-adapter_sd15.bin", min_length=1)


class ImageRequest(BaseModel):
    controlnets: list = [ControlNet]
    disable_text_encoder_3: bool = True
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

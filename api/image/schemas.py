from typing import Literal, Optional

from pydantic import BaseModel


class ImageResponse(BaseModel):
    data: str


class ImageRequest(BaseModel):
    controlnets: list = []
    disable_text_encoder_3: bool = True
    guidance_scale: float = 10.0
    inpainting_full_image: bool = True
    input_image_path: str = ""
    input_mask_path: str = ""
    max_height: int = 2048
    max_width: int = 2048
    model: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    num_inference_steps: int = 25
    output_image_path: str = ""
    prompt: str = "Detailed, 8k, photorealistic"
    seed: int = 42
    strength: float = 0.5
    style_strength: float = 0.5
    style_image_path: str = ""

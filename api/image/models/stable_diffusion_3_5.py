import torch
import os
from utils.utils import get_16_9_resolution
from common.context import Context
from image.models.auto_diffusion import main as auto_diffusion


model_id = "stabilityai/stable-diffusion-3.5-medium"

if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]
    width, height = get_16_9_resolution("720p")

    for guidance_scale in [0.0, 2.0, 5.0]:
        auto_diffusion(
            Context(
                output_image_path=f"../tmp/output/text_to_img_{output_name}_{guidance_scale}.png",
                prompt="A Tidal wave approaching a city, DSLR photo, Detailed, 8k, photorealistic, ",
                negative_prompt="render, artwork, low quality, cartoonish",
                guidance_scale=guidance_scale,
                num_inference_steps=25,
                max_width=width,
                max_height=height,
            ),
            model_id=model_id,
            mode="text_to_image",
        )

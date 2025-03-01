import os

from image.context import ImageContext
from image.models.auto_diffusion import main as auto_diffusion
from image.schemas import ImageRequest
from utils.utils import get_16_9_resolution

model_id = "stabilityai/stable-diffusion-3.5-medium"

if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]
    width, height = get_16_9_resolution("720p")

    for seed in [42, 43, 44]:
        data = ImageRequest(
            model=model_id,
            output_image_path=f"../tmp/output/text_to_img_{output_name}_{seed}.png",
            prompt="A Tidal wave approaching a city, DSLR photo, Detailed, 8k, photorealistic, ",
            negative_prompt="render, artwork, low quality, cartoonish",
            guidance_scale=5,
            num_inference_steps=20,
            max_width=width,
            max_height=height,
        )

        auto_diffusion(
            ImageContext(data),
            mode="text_to_image",
        )

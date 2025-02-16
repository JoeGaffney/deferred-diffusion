import torch
import os
from diffusers import StableDiffusion3Pipeline
from utils.utils import get_16_9_resolution
from utils.diffusers_helpers import diffusers_call, optimize_pipeline
from common.context import Context


pipe = None
model_id = "stabilityai/stable-diffusion-3.5-medium"


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        pipe = optimize_pipeline(pipe)

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    return diffusers_call(pipe, context)


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]
    width, height = get_16_9_resolution("720p")

    for guidance_scale in [0.0, 2.0, 5.0]:
        main(
            Context(
                # input_image_path="../tmp/tornado_v001.JPG",
                output_image_path=f"../tmp/output/text_to_img_{output_name}_{guidance_scale}.png",
                prompt="A Tidal wave approaching a city, DSLR photo, Detailed, 8k, photorealistic, ",
                negative_prompt="render, artwork, low quality, cartoonish",
                guidance_scale=guidance_scale,
                num_inference_steps=25,
                max_width=width,
                max_height=height,
            )
        )

import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from utils.utils import get_16_9_resolution
from common.context import Context
from utils.diffusers_helpers import diffusers_upscale_call

pipe = None
model_id = "stabilityai/stable-diffusion-x4-upscaler"


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision="fp16",
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    return pipe


def main(context: Context):
    pipe = get_pipeline()

    return diffusers_upscale_call(pipe, context)


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    main(
        Context(
            input_image_path="../tmp/tornado_v001.JPG",
            output_image_path=f"../tmp/output/{output_name}.png",
            prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
            guidance_scale=7.5,
            num_inference_steps=10,
        )
    )

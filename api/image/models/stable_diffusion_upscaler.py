from functools import lru_cache
import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from common.context import Context
from utils.diffusers_helpers import upscale_call
from utils.pipeline_helpers import optimize_pipeline


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    return optimize_pipeline(pipe, disable_safety_checker=False)


def main(context: Context, model_id="stabilityai/stable-diffusion-x4-upscaler", mode="upscaler"):
    return upscale_call(get_pipeline(model_id), context, scale=4)


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

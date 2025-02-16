import os
import torch
from diffusers import AutoPipelineForImage2Image
from common.context import Context
from utils.diffusers_helpers import diffusers_image_call, optimize_pipeline

pipe = None


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe = optimize_pipeline(pipe)

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    return diffusers_image_call(pipe, context)


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    for strength in [0.1, 0.35]:

        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                output_image_path=f"../tmp/output/{output_name}_{strength}.png",
                prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
                strength=strength,
            )
        )

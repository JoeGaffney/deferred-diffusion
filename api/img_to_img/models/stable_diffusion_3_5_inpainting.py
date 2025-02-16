import os
import torch
from diffusers import AutoPipelineForInpainting
from utils.diffusers_helpers import diffusers_inpainting_call, optimize_pipeline
from common.context import Context

pipe = None
model_id = "stabilityai/stable-diffusion-3.5-medium"


def get_pipeline():
    global pipe
    if pipe is None:

        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            # text_encoder_3=None,
            # tokenizer_3=None,
        )
        pipe = optimize_pipeline(pipe)

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    return diffusers_inpainting_call(pipe, context)


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    for strength in [0.5, 0.8]:

        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                input_mask_path="../tmp/tornado_v001_mask.png",
                output_image_path=f"../tmp/output/{output_name}_{strength}.png",
                prompt="Trees and roads in the forground, detailed, 8k, photorealistic",
                strength=strength,
                guidance_scale=10,
            )
        )

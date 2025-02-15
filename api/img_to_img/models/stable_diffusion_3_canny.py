import os
import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from utils.diffusers_helpers import diffusers_controlnet_call
from common.context import Context

pipe = None
model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
controlnet_id = "InstantX/SD3-Controlnet-Canny"


def get_pipeline():
    global pipe
    if pipe is None:
        controlnet = SD3ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            controlnet=controlnet,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        pipe.enable_model_cpu_offload()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    return diffusers_controlnet_call(pipe, context)


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    for strength in [0.2, 0.5, 0.75, 1.0]:

        main(
            Context(
                input_image_path="../tmp/canny.png",
                output_image_path=f"../tmp/output/{output_name}_{strength}.png",
                prompt="An eye, Detailed, 8k, photorealistic",
                strength=strength,
                guidance_scale=1.5,
                num_inference_steps=50,
            )
        )

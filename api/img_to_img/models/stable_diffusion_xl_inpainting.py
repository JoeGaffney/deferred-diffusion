import torch
from diffusers import AutoPipelineForInpainting
from common.context import Context
from PIL import Image
import numpy as np

pipe = None


def get_pipeline():
    global pipe
    if pipe is None:

        # Override the safety checker
        def dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)

        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipe.enable_model_cpu_offload()
        pipe.safety_checker = dummy_safety_checker

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image()
    mask = context.load_mask()
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    processed_image = pipe(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
        guidance_scale=context.guidance_scale,
        padding_mask_crop=32,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


if __name__ == "__main__":

    for strength in [0.5, 0.8]:

        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                input_mask_path="../tmp/tornado_v001_mask.png",
                output_image_path=f"../tmp/output/tornado_v001_inpainting_{strength}.png",
                prompt="Trees and roads in the forground, detailed, 8k, photorealistic",
                strength=strength,
                guidance_scale=10,
            )
        )

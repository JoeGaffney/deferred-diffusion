import torch
from diffusers import StableDiffusion3Pipeline
from utils.utils import encode_image_to_latents
from common.context import Context
from PIL import Image
import numpy as np

pipe = None
model_id = "stabilityai/stable-diffusion-3.5-medium"


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image(division=16)  # Load input image
    original_size = image.size  # Get the smaller dimension
    image = image.convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    # Convert the image into latents using the VAE encoder
    latents = encode_image_to_latents(image, pipe.vae)
    # with torch.no_grad():
    #     latents = vae_encode(image, pipe.vae).latent_dist.sample() * 0.18215

    # Perform img2img processing with Stable Diffusion
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    with torch.no_grad():
        proccessed_image = pipe.__call__(
            width=original_size[0],
            height=original_size[1],
            prompt=context.prompt,
            # negative_prompt=context.negative_prompt,
            latents=latents,
            num_inference_steps=context.num_inference_steps,
            guidance_scale=context.guidance_scale,
            # generator=generator,
        ).images[0]

    # proccessed_image = pipe.__call__(
    #     width=original_size[0],
    #     height=original_size[1],
    #     prompt=context.prompt,
    #     negative_prompt=context.negative_prompt,
    #     num_inference_steps=context.num_inference_steps,
    #     guidance_scale=0,
    #     # generator=generator,
    # ).images[0]

    processed_path = context.save_image(proccessed_image)
    return processed_path


if __name__ == "__main__":
    for guidance_scale in [0.0, 7.0]:
        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                output_image_path=f"../tmp/output/tornado_v001_turbo_{guidance_scale}.png",
                prompt="Detailed, 8k, photorealistic, tornado, enhance keep original elements",
                guidance_scale=guidance_scale,
                num_inference_steps=25,
            )
        )

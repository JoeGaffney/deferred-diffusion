import torch
from diffusers import StableDiffusion3Pipeline
from utils.utils import vae_encode
from common.context import Context
from PIL import Image
import numpy as np

pipe = None

model_id = "tensorart/stable-diffusion-3.5-medium-turbo"
# model_id = "stabilityai/stable-diffusion-3.5-large-turbo"


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
        pipe.enable_model_cpu_offload()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image(division=16)  # Load input image
    original_size = image.size  # Get the smaller dimension
    image = image.convert("RGBA")
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    image = pipe.__call__(
        # width=original_size[0],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        num_inference_steps=context.num_inference_steps,
        guidance_scale=0,
        # generator=generator,
    ).images[0]
    context.save_image(image)
    return
    # image = np.array(image).astype(np.float32) / 255.0  # Normalize [0, 1]
    # image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to("cuda")  # (B, C, H, W)
    # image = 2 * image - 1  # Normalize to [-1, 1]

    # # Convert to float16 (or bfloat16) for compatibility with model
    # image = image.to(torch.float16)  # Ensure input is float16
    # image = np.array(image).astype(np.float16)
    # # if image.ndim == 3:  # Check if the array is 3D (height, width, channels)
    # #     image = np.concatenate(
    # #         [image, np.ones((image.shape[0], image.shape[1], 1), dtype=image.dtype)], axis=-1
    # #     )  # Add an alpha channel (RGBA)
    # if image.ndim == 3:
    #     image = np.expand_dims(image, axis=0)  # Adds a batch dimension (1, height, width, channels)

    # Convert the image into latents using the VAE encoder
    with torch.no_grad():
        latents = vae_encode(image, pipe.vae).latent_dist.sample()
        # print(latents)
        # latents = pipe.vae.encode(image).latent_dist.sample() * 0.18215  # Encoding step

    # Perform img2img processing with Stable Diffusion
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    with torch.no_grad():
        output_latents = pipe(
            width=original_size[0],
            height=original_size[1],
            prompt=context.prompt,
            negative_prompt=context.negative_prompt,
            # latents=latents,
            num_inference_steps=4,
            guidance_scale=7.5,
            generator=generator,
        ).images[0]

    processed_path = context.save_image(output_latents)
    return processed_path

    # # Decode the processed latents back to an image
    # with torch.no_grad():
    #     print(output_latents)
    #     decoded = pipe.vae.decode(output_latents).sample

    # print(decoded)
    # # # Normalize decoded image back to [0, 1] and convert to uint8
    # # decoded = (decoded / 2 + 0.5).clamp(0, 1)
    # # decoded_image = (decoded * 255).cpu().numpy().astype(np.uint8).squeeze(0)
    # # decoded_image = np.transpose(decoded_image, (1, 2, 0))  # Convert to (H, W, C)
    # # decoded_image = Image.fromarray(decoded_image)  # Convert to PIL Image

    # # Save the final processed image
    # processed_path = context.save_image(output_latents)
    # return processed_path


if __name__ == "__main__":
    for strength in [0.35, 0.5]:
        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                output_image_path=f"../tmp/output/tornado_v001_turbo_{strength}.png",
                prompt="Detailed, 8k, photorealistic, tornado, enhance keep original elements",
                strength=strength,
                num_inference_steps=4,
            )
        )

import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from common.context import Context

pipe = None
model_id = "tensorart/stable-diffusion-3.5-medium-turbo"
# model_id = "stabilityai/stable-diffusion-3.5-large-turbo" # too slow


def get_pipeline():
    global pipe
    if pipe is None:
        pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    return pipe


def main(context: Context):
    pipe = get_pipeline()
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    processed_image = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
        guidance_scale=context.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


if __name__ == "__main__":

    for strength in [0.2, 0.5, 0.75, 1.0]:

        main(
            Context(
                input_image_path="../tmp/tornado_v001.JPG",
                output_image_path=f"../tmp/output/tornado_v001_img_to_img_turbo_{strength}.png",
                prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
                strength=strength,
                guidance_scale=0.0,
            )
        )

import torch
from diffusers import AutoPipelineForImage2Image
from common.context import Context

pipe = None


def get_pipeline():
    global pipe
    if pipe is None:

        # Override the safety checker
        def dummy_safety_checker(images, **kwargs):
            return images, [False] * len(images)

        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
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
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    processed_image = pipe(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image, with_timestamp=False)
    return processed_path


if __name__ == "__main__":

    for strength in [0.1, 0.35]:
        main(
            Context(
                image="space_v001.jpg",
                output_name=f"space_{strength}",
                strength=strength,
                prompt="Detailed, 8k, add a spaceship, higher contrast, enchance keep original elements",
            )
        )
        main(
            Context(
                image="tornado_v001.jpg",
                output_name=f"tornado_{strength}",
                strength=strength,
                prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
                size_multiplier=1.0,
            )
        )
        main(
            Context(
                image="earth_quake_v001.jpg",
                output_name=f"earth_quake_{strength}",
                strength=strength,
                prompt="Detailed, 8k, photorealistic",
            )
        )
        main(Context(image="elf_v001.jpg", strength=0.1, prompt="Detailed, 8k, photorealistic", size_multiplier=1.0))

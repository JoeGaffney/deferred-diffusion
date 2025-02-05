import math
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

from api.common.context import Context
from api.utils import device_info


# Load Stable Diffusion model
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.enable_model_cpu_offload()


# Override the safety checker
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


pipe.safety_checker = dummy_safety_checker


def main(context: Context):
    # Decode the base64 image data
    input_image = Image.open(context.get_input_image_path()).convert("RGB")

    # Resize the image
    width = input_image.size[0] * context.size_multiplier
    height = input_image.size[1] * context.size_multiplier
    width = math.ceil(width / 8) * 8
    height = math.ceil(height / 8) * 8
    input_image = input_image.resize((width, height))

    # Flip the image horizontally
    # input_image = input_image.transpose(Image.FLIP_TOP_BOTTOM)

    # Run Stable Diffusion (e.g., text-to-image or image-to-image)
    processed_image = pipe(context.prompt, image=input_image, strength=context.strength).images[0]

    # Save the processed image with a timestamp
    processed_image_path = context.save_image(processed_image)


if __name__ == "__main__":

    for strength in [0.1, 0.35]:
        main(
            Context(
                image="space_v001.jpg",
                strength=strength,
                prompt="Detailed, 8k, add a spaceship, higher contrast, enchance keep original elements",
            )
        )
        main(
            Context(
                image="tornado_v001.jpg",
                strength=strength,
                prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
                size_multiplier=1.0,
            )
        )
        main(
            Context(
                image="earth_quake_v001.jpg",
                strength=strength,
                prompt="Detailed, 8k, photorealistic",
                size_multiplier=1.0,
            )
        )
        main(Context(image="elf_v001.jpg", strength=0.1, prompt="Detailed, 8k, photorealistic", size_multiplier=1.0))

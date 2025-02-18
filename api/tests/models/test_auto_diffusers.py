from utils.utils import get_16_9_resolution
from common.context import Context
from models.auto_diffusion import main
import os


def validation_tests(
    output_name,
    model_ids=[
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/stable-diffusion-3-medium-diffusers",
        "stabilityai/stable-diffusion-3.5-medium",
    ],
    controlnets=[],
    prompt="Detailed, 8k, DSLR photo, photorealistic, Eye, enchance keep original elements",
):
    width, height = get_16_9_resolution("540p")

    prompt = "Detailed, 8k, DSLR photo, photorealistic, Eye, enchance keep original elements"

    for model_id in model_ids:
        for mode in ["text_to_image", "img_to_img", "img_to_img_inpainting"]:
            model_id_nice = model_id.replace("/", "_")

            main(
                Context(
                    model=model_id,
                    input_image_path="../tmp/tornado_v001.JPG",
                    input_mask_path="../tmp/tornado_v001_mask.png",
                    output_image_path=f"../tmp/output/{model_id_nice}/{output_name}_{mode}.png",
                    prompt=prompt,
                    strength=0.5,
                    guidance_scale=7.5,
                    max_width=width,
                    max_height=height,
                    controlnets=controlnets,
                ),
                model_id=model_id,
                mode=mode,
            )


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    controlnet_a = {
        "model": "InstantX/SD3-Controlnet-Canny",
        "input_image": "../tmp/canny.png",
        "conditioning_scale": "0.5",
    }
    controlnet_b = {
        "model": "diffusers/controlnet-canny-sdxl-1.0",
        "input_image": "../tmp/canny.png",
        "conditioning_scale": "0.5",
    }

    validation_tests(
        output_name,
        model_ids=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
        ],
        controlnets=[],
        prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enchance keep original elements",
    )

    validation_tests(
        output_name + "_controlnets",
        model_ids=[
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
        controlnets=[controlnet_b, controlnet_b],
    )

    validation_tests(
        output_name + "_controlnets",
        model_ids=[
            "stabilityai/stable-diffusion-3-medium-diffusers",
            "stabilityai/stable-diffusion-3.5-medium",
        ],
        controlnets=[controlnet_a, controlnet_a],
    )

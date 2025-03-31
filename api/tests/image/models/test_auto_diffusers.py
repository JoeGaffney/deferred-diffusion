import os

import pytest
from image.context import ImageContext
from image.models.auto_diffusion import main
from image.schemas import ImageRequest
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["text_to_image", "img_to_img", "img_to_img_inpainting"]

MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-medium",
]

MODEL_CONTROLNET_MAPPING = {
    "stabilityai/stable-diffusion-xl-base-1.0": [
        {
            "model": "diffusers/controlnet-canny-sdxl-1.0",
            "input_image": "../test_data/canny_v001.png",
            "conditioning_scale": "0.5",
        }
    ],
    "stabilityai/stable-diffusion-3-medium-diffusers": [
        {
            "model": "InstantX/SD3-Controlnet-Canny",
            "input_image": "../test_data/canny_v001.png",
            "conditioning_scale": "0.5",
        }
    ],
    "stabilityai/stable-diffusion-3.5-medium": [
        {
            "model": "InstantX/SD3-Controlnet-Canny",
            "input_image": "../test_data/canny_v001.png",
            "conditioning_scale": "0.5",
        }
    ],
}


@pytest.mark.parametrize("mode", ["text_to_image", "img_to_img", "img_to_img_inpainting"])
@pytest.mark.parametrize("model_id", MODELS)
def test(model_id, mode):
    """Test models WITHOUT control nets."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="../test_data/color_v001.jpeg",
                input_mask_path="../test_data/mask_v001.png",
                output_image_path=output_name,
                prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enhance keep original elements",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
            )
        ),
        mode=mode,
    )

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."


@pytest.mark.parametrize("mode", ["text_to_image", "img_to_img", "img_to_img_inpainting"])
@pytest.mark.parametrize("model_id", MODELS)
def test_with_controlnets(model_id, mode):
    """Test models WITH control nets."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_with_controlnets.png"
    width, height = get_16_9_resolution("540p")
    controlnets = MODEL_CONTROLNET_MAPPING[model_id]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="../test_data/color_v001.jpeg",
                input_mask_path="../test_data/mask_v001.png",
                output_image_path=output_name,
                prompt="Detailed, 8k, DSLR photo, photorealistic, eye",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=controlnets,
            )
        ),
        mode=mode,
    )

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

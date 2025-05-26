import os

import pytest

from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ControlNetSchema, ImageRequest, IpAdapterModel
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution

MODES = ["text_to_image", "img_to_img", "img_to_img_inpainting"]
MODELS = ["sdxl", "RealVisXL"]
MODELS = ["RealVisXL"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models(model_id, mode):
    output_name = setup_output_file(model_id, mode)
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../assets/color_v001.jpeg"),
                mask=image_to_base64("../assets/mask_v001.png"),
                prompt="tornado on farm feild, enhance keep original elements, Detailed, 8k, DSLR photo, photorealistic",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
            )
        ),
        mode=mode,
    )

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", MODELS)
def test_models_with_control_nets(model_id, mode):
    """Test models with control nets."""
    output_name = setup_output_file(model_id, mode, "_controlnet")
    width, height = get_16_9_resolution("540p")
    controlnets = [
        ControlNetSchema(model="canny", image=image_to_base64("../assets/canny_v001.png"), conditioning_scale=0.5)
    ]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../assets/color_v001.jpeg"),
                mask=image_to_base64("../assets/mask_v001.png"),
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

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", MODELS)
def test_style(model_id, mode):
    """Test models with style adapter."""
    output_name = setup_output_file(model_id, mode, "_style_adapter")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../assets/color_v001.jpeg"),
                prompt="a cat, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=[
                    IpAdapterModel(image=image_to_base64("../assets/style_v001.jpeg"), model="style", scale=0.5)
                ],
            )
        ),
        mode=mode,
    )

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", MODELS)
def test_face(model_id, mode):
    """Test models with face adapter."""
    output_name = setup_output_file(model_id, mode, "_face_adapter")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                prompt="a man walking, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                ip_adapters=[
                    IpAdapterModel(image=image_to_base64("../assets/style_v001.jpeg"), model="style", scale=0.5),
                    IpAdapterModel(image=image_to_base64("../assets/face_v001.jpeg"), model="face", scale=0.5),
                ],
            )
        ),
        mode=mode,
    )

    save_image_and_assert_file_exists(result, output_name)

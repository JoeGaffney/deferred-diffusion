import os

import pytest

from images.context import ImageContext
from images.models.sdxl import main
from images.schemas import ImageRequest, ModelName, References
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution

MODES = ["text_to_image", "img_to_img", "img_to_img_inpainting"]
model: ModelName = "sd-xl"


# @pytest.mark.parametrize("mode", MODES)
# def test_models(mode):
#     output_name = setup_output_file(model, mode)
#     width, height = get_16_9_resolution("540p")

#     result = main(
#         ImageContext(
#             ImageRequest(
#                 model=model,
#                 image=None if mode == "text_to_image" else image_to_base64("../assets/color_v001.jpeg"),
#                 mask=None if mode == "img_to_img_inpainting" else image_to_base64("../assets/mask_v001.png"),
#                 prompt="tornado on farm feild, enhance keep original elements, Detailed, 8k, DSLR photo, photorealistic",
#                 strength=0.5,
#                 guidance_scale=5,
#                 width=width,
#                 height=height,
#                 controlnets=[],
#             )
#         )
#     )

#     save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
def test_models_with_canny(mode):
    output_name = setup_output_file(model, mode, "_canny")
    width, height = get_16_9_resolution("540p")
    references = [
        References(
            mode="canny",
            image=image_to_base64("../assets/canny_v001.png"),
            strength=0.5,
        )
    ]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Detailed, 8k, DSLR photo, photorealistic, eye",
                strength=0.5,
                guidance_scale=5,
                width=width,
                height=height,
                references=references,
            )
        ),
    )

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
def test_style(mode):
    """Test models with style adapter."""
    output_name = setup_output_file(model, mode, "_style")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a cat, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                width=width,
                height=height,
                references=[
                    References(
                        mode="style",
                        image=image_to_base64("../assets/style_v001.jpeg"),
                        strength=0.5,
                    )
                ],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("mode", ["text_to_image"])
def test_face(mode):
    """Test models with face adapter."""
    output_name = setup_output_file(model, mode, "_face")
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="a man walking, masterpiece, best quality, high quality",
                negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
                strength=0.75,
                guidance_scale=5,
                width=width,
                height=height,
                references=[
                    References(
                        mode="style",
                        image=image_to_base64("../assets/style_v001.jpeg"),
                        strength=0.5,
                    ),
                    References(
                        mode="face",
                        image=image_to_base64("../assets/face_v001.jpeg"),
                        strength=0.5,
                    ),
                ],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)

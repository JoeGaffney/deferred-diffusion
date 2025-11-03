from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from tests.images.helpers import main
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)

models: List[ModelName] = ["real-esrgan-x4"]


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    output_name = setup_output_file(model, "upscale_image")

    result = main(
        ImageContext(
            model,
            ImageRequest(
                prompt="Change to night time and add rain and lighting",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)

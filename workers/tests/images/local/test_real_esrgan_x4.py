from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from tests.images.helpers import main
from tests.utils import asset_outputs_exists, image_to_base64

models: List[ModelName] = ["real-esrgan-x4"]


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    output_name = "upscale_image"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Change to night time and add rain and lighting",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)

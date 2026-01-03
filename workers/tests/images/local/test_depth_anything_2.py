from typing import List

import pytest

from images.context import ImageContext
from images.local.depth_anything_2 import main
from images.schemas import ImageRequest, ModelName
from tests.utils import asset_outputs_exists, image_to_base64
from utils.utils import get_16_9_resolution

models: List[ModelName] = ["depth-anything-2"]


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    width, height = get_16_9_resolution("540p")
    output_name = "depth"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                image=image_to_base64("../assets/color_v001.jpeg"),
                prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enhance keep original elements",
                strength=0.5,
                width=width,
                height=height,
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)

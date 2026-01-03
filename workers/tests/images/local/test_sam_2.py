from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from tests.images.helpers import main
from tests.utils import asset_outputs_exists, image_to_base64

models: List[ModelName] = ["sam-2"]


@pytest.mark.parametrize("model", models)
def test_segmentation(model):
    output_name = "segmentation"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="house and trees",
                strength=0.5,
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)


@pytest.mark.parametrize("model", models)
def test_segmentation_alt(model):
    output_name = "segmentation_alt"

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="Rabbit",
                strength=0.5,
            ),
            task_id=output_name,
        )
    )

    asset_outputs_exists(result)

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

models: List[ModelName] = ["sam-3-image", "sam-2-image"]


@pytest.mark.parametrize("model", models)
def test_segmentation(model):
    output_name = setup_output_file(model, "segmentation")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="house and trees",
                strength=0.5,
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)


@pytest.mark.parametrize("model", models)
def test_segmentation_alt(model):
    output_name = setup_output_file(model, "segmentation_alt")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="Rabbit",
                strength=0.5,
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)

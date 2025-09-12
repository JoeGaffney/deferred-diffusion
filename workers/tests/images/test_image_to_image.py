from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from images.tasks import model_router_main as main
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)

MODES = ["image_to_image"]
models: List[ModelName] = ["sd-xl", "sd-3", "flux-1", "flux-kontext-1", "qwen-image"]
models_external: List[ModelName] = [
    "flux-1-1-pro",
    "flux-kontext-1-pro",
    "gpt-image-1",
    "runway-gen4-image",
]
models.extend(models_external)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", models)
def test_image_to_image(model, mode):
    output_name = setup_output_file(model, mode)

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Change to night time and add rain and lighting",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)

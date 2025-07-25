from typing import List

import pytest

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from images.tasks import external_model_router_main as main
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)

MODES = ["image_to_image"]
models: List[ModelName] = [
    "external-flux-1-1",
    "external-flux-kontext",
    "external-gpt-image-1",
    "external-runway-gen4-image",
]


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
                num_inference_steps=10,
                guidance_scale=3.5,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)
    free_gpu_memory()

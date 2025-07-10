from typing import List

import pytest

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from images.tasks import model_router_main as main
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)

MODES = ["inpainting"]
models: List[ModelName] = ["sd-xl", "sd-3", "flux-1"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", models)
def test_image_to_image(model, mode):
    output_name = setup_output_file(model, mode)

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Photorealistic landscape of an elven castle, inspired by lord of the rings, highly detailed, 8k",
                strength=0.5,
                image=image_to_base64("../assets/inpaint.png"),
                mask=image_to_base64("../assets/inpaint_mask.png"),
                num_inference_steps=25,
                guidance_scale=3.5,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)
    free_gpu_memory()

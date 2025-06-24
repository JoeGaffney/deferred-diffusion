import pytest

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.external_models.flux_kontext import main
from images.schemas import ImageRequest
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["image_to_image"]
MODELS = ["flux-kontext-pro"]


# @pytest.mark.skip(reason="Slow test")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models(model_id, mode):
    output_name = output_name = setup_output_file(model_id, mode)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                prompt="Change to night time and add rain and lighting",
                strength=0.5,
                image=image_to_base64("../assets/color_v001.jpeg"),
                num_inference_steps=15,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)
    free_gpu_memory()

import pytest

from images.context import ImageContext
from images.local.depth_anything_2 import main
from images.schemas import ImageRequest, ModelName
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution

model: ModelName = "depth-anything-2"


@pytest.mark.parametrize("mode", ["depth"])
def test_models(mode):
    output_name = setup_output_file(model, mode)
    width, height = get_16_9_resolution("540p")

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
        )
    )

    save_image_and_assert_file_exists(result, output_name)

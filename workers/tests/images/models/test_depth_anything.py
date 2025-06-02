import pytest

from images.context import ImageContext
from images.models.depth_anything import main
from images.schemas import ImageRequest
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)
from utils.utils import get_16_9_resolution


@pytest.mark.parametrize("mode", ["depth"])
def test_models(mode):
    """Test models."""
    model_id = "depth-anything"
    output_name = setup_output_file(model_id, mode)
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=image_to_base64("../assets/color_v001.jpeg"),
                prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enhance keep original elements",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)

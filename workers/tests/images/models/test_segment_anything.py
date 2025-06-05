import pytest

from images.context import ImageContext
from images.models.segment_anything import main
from images.schemas import ImageRequest
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)


@pytest.mark.parametrize("mode", ["mask"])
def test_models(mode):
    """Test models."""
    model_id = "segment-anything"
    output_name = setup_output_file(model_id, mode)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="Person, house, tree, flowers",
                strength=0.5,
                guidance_scale=5,
                controlnets=[],
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)

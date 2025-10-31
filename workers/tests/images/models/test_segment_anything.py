import pytest

from images.context import ImageContext
from images.models.segment_anything import main
from images.schemas import ImageRequest, ModelName
from tests.utils import (
    image_to_base64,
    save_image_and_assert_file_exists,
    setup_output_file,
)

model: ModelName = "segment-anything-2"


@pytest.mark.parametrize("mode", ["mask"])
def test_models(mode):
    """Test models."""
    output_name = setup_output_file(model, mode)

    result = main(
        ImageContext(
            model,
            ImageRequest(
                image=image_to_base64("../assets/style_v001.jpeg"),
                prompt="Person, house, tree, flowers",
                strength=0.5,
            ),
        )
    )

    save_image_and_assert_file_exists(result, output_name)

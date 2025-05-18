import os

import pytest
from PIL import Image

from images.context import ImageContext
from images.models.segment_anything import main
from images.schemas import ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import ensure_path_exists


@pytest.mark.parametrize("mode", ["mask"])
def test_models(mode):
    """Test models."""
    model_id = "segment-anything"

    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}.png"

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

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
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

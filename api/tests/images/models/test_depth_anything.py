import os

import pytest

from images.context import ImageContext
from images.models.depth_anything import main
from images.schemas import ImageRequest
from utils.utils import get_16_9_resolution


@pytest.mark.parametrize("mode", ["depth"])
def test_models(mode):
    """Test models."""
    model_id = "depth-anything"

    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="../test_data/color_v001.jpeg",
                input_mask_path="",
                output_image_path=output_name,
                prompt="Detailed, 8k, DSLR photo, photorealistic, tornado, enhance keep original elements",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
            )
        ),
        mode=mode,
    )

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

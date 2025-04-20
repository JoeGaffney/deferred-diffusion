import os

import pytest

from image.context import ImageContext
from image.models.auto_diffusion import main
from image.schemas import ImageRequest
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["text_to_image", "img_to_img", "img_to_img_inpainting"]

MODELS = [
    "sdxl",
    "sd3",
    "sd3.5",
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models(model_id, mode):
    """Test models."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    main(
        ImageContext(
            ImageRequest(
                model=model_id,
                input_image_path="" if mode == "text_to_image" else "../test_data/color_v001.jpeg",
                input_mask_path="../test_data/mask_v001.png",
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

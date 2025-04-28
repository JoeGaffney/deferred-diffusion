import os

import pytest

from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
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
                image=None if mode == "text_to_image" else image_to_base64("../test_data/color_v001.jpeg"),
                mask=image_to_base64("../test_data/mask_v001.png"),
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

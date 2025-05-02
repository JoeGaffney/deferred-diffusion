import os

import pytest
from PIL import Image

from common.logger import logger
from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ControlNetSchema, ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import (
    ensure_path_exists,
    get_16_9_resolution,
    get_gpu_memory_usage_pretty,
)

# Define constants
MODES = ["text_to_image", "img_to_img"]

# Change the MODELS list to use the correct keys from ImageRequest.model Literal
MODELS = ["sdxl", "sd3", "sd3.5"]

# Update the mapping to use these keys
MODEL_CONTROLNET_MAPPING = {
    "sdxl": [
        ControlNetSchema(model="canny", image=image_to_base64("../test_data/canny_v001.png"), conditioning_scale=0.5)
    ],
    "sd3": [
        ControlNetSchema(model="canny", image=image_to_base64("../test_data/canny_v001.png"), conditioning_scale=0.5)
    ],
    "sd3.5": [
        ControlNetSchema(model="canny", image=image_to_base64("../test_data/canny_v001.png"), conditioning_scale=0.5)
    ],
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models_with_control_nets(model_id, mode):
    """Test models with control nets."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_cn_canny.png"
    width, height = get_16_9_resolution("540p")
    controlnets = MODEL_CONTROLNET_MAPPING[model_id]

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../test_data/color_v001.jpeg"),
                mask=image_to_base64("../test_data/mask_v001.png"),
                prompt="Detailed, 8k, DSLR photo, photorealistic, eye",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=controlnets,
            )
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

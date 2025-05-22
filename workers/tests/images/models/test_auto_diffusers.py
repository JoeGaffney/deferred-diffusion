import os

import pytest
from PIL import Image

from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import ensure_path_exists, get_16_9_resolution

# Define constants
MODES = ["text_to_image", "img_to_img", "img_to_img_inpainting"]
MODELS = ["sd1.5", "sdxl", "sd3"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models(model_id, mode, target_precision):
    """Test models."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=None if mode == "text_to_image" else image_to_base64("../assets/color_v001.jpeg"),
                mask=image_to_base64("../assets/mask_v001.png"),
                prompt="tornado on farm feild, enhance keep original elements, Detailed, 8k, DSLR photo, photorealistic",
                strength=0.5,
                guidance_scale=5,
                max_width=width,
                max_height=height,
                controlnets=[],
                target_precision=target_precision,
            )
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."


@pytest.mark.parametrize("mode", ["text_to_image"])
@pytest.mark.parametrize("model_id", ["flux-schnell", "sd3.5"])
@pytest.mark.parametrize("target_precision", [4, 8, 16])
def test_precision_models(model_id, mode, target_precision):
    """Test models."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_precsion{target_precision}.png"
    width, height = get_16_9_resolution("540p")

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                prompt="tornado on farm feild, enhance keep original elements, Detailed, 8k, DSLR photo, photorealistic",
                strength=0.5,
                guidance_scale=3.5,
                max_width=width,
                max_height=height,
                controlnets=[],
                target_precision=target_precision,
                num_inference_steps=15,
            )
        ),
        mode=mode,
    )

    if isinstance(result, Image.Image):
        ensure_path_exists(output_name)
        result.save(output_name)

    # Check if output file exists
    assert os.path.exists(output_name), f"Output file {output_name} was not created."

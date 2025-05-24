import os

import pytest
from PIL import Image

from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import ensure_path_exists, get_16_9_resolution

# Define constants
MODES = ["img_to_img"]
MODELS = ["HiDream"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("target_precision", [4, 8])
def test_models(model_id, mode, target_precision):
    """Test models."""
    output_name = f"../tmp/output/{model_id.replace('/', '_')}/{mode}_precsion{target_precision}.png"

    # Delete existing file if it exists
    if os.path.exists(output_name):
        os.remove(output_name)

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                image=image_to_base64("../assets/color_v002.png"),
                prompt="Editing Instruction: Convert the image into a Ghibli style. Target Image Description: A person playing a guitar, depicted in a Ghibli style against a plain background.",
                guidance_scale=5,
                controlnets=[],
                num_inference_steps=10,
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

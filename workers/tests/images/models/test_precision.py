import os

import pytest
from PIL import Image

from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest
from tests.utils import image_to_base64, optional_image_to_base64
from utils.utils import ensure_path_exists, get_16_9_resolution

# Define constants
MODES = ["text_to_image"]
MODELS = ["HiDream"]
# @pytest.mark.parametrize("model_id", ["flux-schnell", "sd3.5"])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("target_precision", [4, 8])
def test_models(model_id, mode, target_precision):
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
                prompt="A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature.",
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

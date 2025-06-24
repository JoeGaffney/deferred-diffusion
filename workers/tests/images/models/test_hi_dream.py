import pytest

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.models.auto_diffusion import main
from images.schemas import ImageRequest
from tests.utils import save_image_and_assert_file_exists, setup_output_file
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["text_to_image"]
MODELS = ["HiDream"]


# @pytest.mark.skip(reason="Slow test")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model_id", MODELS)
def test_models(model_id, mode):
    output_name = output_name = setup_output_file(model_id, mode)
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model_id,
                prompt="A serene scene of a woman lying on lush green grass in a sunlit meadow. She has long flowing hair spread out around her, eyes closed, with a peaceful expression on her face. She's wearing a light summer dress that gently ripples in the breeze. Around her, wildflowers bloom in soft pastel colors, and sunlight filters through the leaves of nearby trees, casting dappled shadows. The mood is calm, dreamy, and connected to nature.",
                strength=0.5,
                guidance_scale=3.5,
                width=width,
                height=height,
                controlnets=[],
                num_inference_steps=15,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)
    free_gpu_memory()

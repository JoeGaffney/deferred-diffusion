import pytest

from common.memory import free_gpu_memory
from images.context import ImageContext
from images.models.flux import main
from images.schemas import ImageRequest, ModelName
from tests.utils import save_image_and_assert_file_exists, setup_output_file
from utils.utils import get_16_9_resolution

# Define constants
MODES = ["text_to_image"]
model: ModelName = "flux-1"


@pytest.mark.parametrize("mode", MODES)
def test_models(model_id, mode):
    output_name = setup_output_file(mode, mode)
    width, height = get_16_9_resolution("540p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
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

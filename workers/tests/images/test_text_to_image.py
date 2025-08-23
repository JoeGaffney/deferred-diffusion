from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from images.tasks import model_router_main as main
from tests.utils import save_image_and_assert_file_exists, setup_output_file
from utils.utils import get_16_9_resolution

MODES = ["text_to_image"]
models: List[ModelName] = ["sd-xl", "sd-3", "flux-1", "flux-1-krea", "qwen-image"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("model", models)
def test_text_to_image(model, mode):

    output_name = setup_output_file(model, mode)
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

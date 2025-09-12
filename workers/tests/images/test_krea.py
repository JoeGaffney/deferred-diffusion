from typing import List

import pytest

from images.context import ImageContext
from images.schemas import ImageRequest, ModelName
from images.tasks import model_router_main as main
from tests.utils import save_image_and_assert_file_exists, setup_output_file
from utils.utils import get_16_9_resolution

MODES = ["text_to_image"]
# models: List[ModelName] = ["sd-xl", "sd-3", "flux-1", "flux-1-krea", "qwen-image"]
# models_external: List[ModelName] = [
#     "flux-1-1-pro",
#     "gpt-image-1",
#     "runway-gen4-image",
#     "google-gemini-2-5",
# ]
# models.extend(models_external)
models = ["google-gemini-2-5", "flux-1-1-pro", "flux-1-krea", "qwen-image"]


@pytest.mark.parametrize("seed", range(42, 46))
@pytest.mark.parametrize("model", models)
def test_text_to_image(model, seed):

    output_name = setup_output_file(model, "test", suffix=f"{seed}")
    width, height = get_16_9_resolution("720p")

    result = main(
        ImageContext(
            ImageRequest(
                model=model,
                prompt="Sperm whale (Moby Dick) diving deep in the dark ocean, giant squid looming in the background, photorealistic, DSLR-style photography, cinematic lighting, nature documentary style, deep ocean atmosphere",
                width=width,
                height=height,
                seed=seed,
            )
        )
    )

    save_image_and_assert_file_exists(result, output_name)

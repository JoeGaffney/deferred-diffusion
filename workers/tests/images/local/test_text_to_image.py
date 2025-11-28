from typing import List

import pytest

from images.schemas import ModelName
from tests.images.helpers import text_to_image, text_to_image_alt

models: List[ModelName] = [
    "sd-xl",
    "flux-1",
    "qwen-image",
    "flux-2",
]


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("model", models)
def test_text_to_image(model, seed):
    text_to_image(model, seed=seed)

    # Uncomment to test alternative image prompt
    # text_to_image_alt(model, seed=seed)

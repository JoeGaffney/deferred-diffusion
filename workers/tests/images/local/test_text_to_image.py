from typing import List, Literal

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    image_to_image_alt,
    inpainting,
    inpainting_alt,
    references_canny,
    references_depth,
    references_face,
    references_style,
    text_to_image,
    text_to_image_alt,
)

models: List[ModelName] = [
    "sd-xl",
    "flux-1",
    "qwen-image",
    "flux-2",
]


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("model", models)
def test_text_to_image(model, seed):
    text_to_image_alt(model, seed=seed)
    # text_to_image_alt(model, seed=seed)

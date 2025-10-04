from typing import List, Literal

import pytest
from helpers import (
    image_to_image,
    inpainting,
    inpainting_alt,
    references_canny,
    references_face,
    references_style,
    text_to_image,
)

from images.schemas import ModelName

models: List[ModelName] = ["qwen-image"]


# @pytest.mark.parametrize("model", models)
# def test_text_to_image(model):
#     text_to_image(model)


# @pytest.mark.parametrize("model", models)
# def test_inpainting(model):
#     inpainting(model)


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    image_to_image(model)

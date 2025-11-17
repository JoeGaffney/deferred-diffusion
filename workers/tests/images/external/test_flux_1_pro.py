from typing import List

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    inpainting,
    inpainting_alt,
    text_to_image,
)

models: List[ModelName] = ["flux-1-pro"]


@pytest.mark.parametrize("model", models)
def test_text_to_image(model):
    text_to_image(model)


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    image_to_image(model)


@pytest.mark.parametrize("model", models)
def test_inpainting(model):
    inpainting(model)


@pytest.mark.parametrize("model", models)
def test_inpainting_alt(model):
    inpainting_alt(model)

from typing import List, Literal

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    image_to_image_alt,
    inpainting,
    inpainting_alt,
    text_to_image_alt,
)

models: List[ModelName] = ["sd-xl"]


@pytest.mark.parametrize("model", models)
def test_text_to_image_alt(model):
    text_to_image_alt(model)


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    image_to_image(model)


@pytest.mark.parametrize("model", models)
def test_image_to_image_alt(model):
    image_to_image_alt(model)


@pytest.mark.parametrize("model", models)
def test_inpainting(model):
    inpainting(model)


@pytest.mark.parametrize("model", models)
def test_inpainting_alt(model):
    inpainting_alt(model)

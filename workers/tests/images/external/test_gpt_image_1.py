from typing import List

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    inpainting,
    inpainting_alt,
    references_canny,
    references_face,
    references_style,
    text_to_image,
    text_to_image_alt,
)

models: List[ModelName] = ["gpt-image-1"]


@pytest.mark.parametrize("model", models)
def test_text_to_image(model):
    text_to_image(model)


@pytest.mark.parametrize("model", models)
def test_text_to_image_alt(model):
    text_to_image_alt(model)


@pytest.mark.parametrize("model", models)
def test_image_to_image(model):
    image_to_image(model)


@pytest.mark.parametrize("model", models)
def test_references_canny(model):
    references_canny(model)


@pytest.mark.parametrize("model", models)
def test_references_face(model):
    references_face(model)


@pytest.mark.parametrize("model", models)
def test_references_style(model):
    references_style(model)


@pytest.mark.parametrize("model", models)
def test_inpainting(model):
    inpainting(model)


@pytest.mark.parametrize("model", models)
def test_inpainting_alt(model):
    inpainting_alt(model)

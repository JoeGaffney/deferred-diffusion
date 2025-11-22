from typing import List

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    references_canny,
    references_face,
    references_style,
    text_to_image,
    text_to_image_alt,
)

models: List[ModelName] = ["google-gemini-3"]


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

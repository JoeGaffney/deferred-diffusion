from typing import List

import pytest

from images.schemas import ModelName
from tests.images.helpers import (
    image_to_image,
    references_canny,
    references_face,
    references_style,
    text_to_image,
)

models: List[ModelName] = ["gemini-2"]


@pytest.mark.parametrize("model", models)
def test_text_to_image(model):
    text_to_image(model)


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

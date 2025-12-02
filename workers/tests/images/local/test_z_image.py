from typing import List, Literal

import pytest

from images.schemas import ModelName
from tests.images.helpers import text_to_image, text_to_image_alt

models: List[ModelName] = ["z-image"]


@pytest.mark.basic
@pytest.mark.parametrize("model", models)
def test_text_to_image(model):
    text_to_image(model)


@pytest.mark.parametrize("model", models)
def test_text_to_image_alt(model):
    text_to_image_alt(model)

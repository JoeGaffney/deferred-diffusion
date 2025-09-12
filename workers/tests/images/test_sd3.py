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

models: List[ModelName] = ["sd-3"]


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


# @pytest.mark.parametrize("model", models)
# def test_references_canny(model):
#     references_canny(model)


# @pytest.mark.parametrize("model", models)
# def test_references_face(model):
#     references_face(model)


# @pytest.mark.parametrize("model", models)
# def test_references_style(model):
#     references_style(model)

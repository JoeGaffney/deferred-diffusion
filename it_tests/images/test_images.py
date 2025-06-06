import json
import os
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import (
    images_create,
    images_get,
    images_get_workflow_schema,
)
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models.comfy_workflow_response import ComfyWorkflowResponse
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from utils import image_to_base64, save_image_and_assert_file_exists

model = ImageRequestModel("sd1.5")
output_dir = "../tmp/output/it-tests/images"


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DEF_DIF_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DEF_DIF_API_KEY", ""),
    )


def create_image(api_client, body: ImageRequest) -> UUID:
    """Helper function to create an image and return its ID."""

    response = images_create.sync_detailed(client=api_client, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


def test_create_image(api_client):
    body = ImageRequest(prompt="A beautiful mountain landscape", model=model, max_width=512, max_height=512)
    image_id = create_image(api_client, body)
    assert isinstance(image_id, UUID)


def test_get_workflor_schema(api_client):
    """Test to ensure the workflow schema can be retrieved."""
    response = images_get_workflow_schema.sync_detailed(client=api_client)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ComfyWorkflowResponse)
    print(response.parsed.to_dict())


def test_get_image(api_client):
    body = ImageRequest(prompt="A beautiful mountain landscape", model=model, max_width=512, max_height=512)
    image_id = create_image(api_client, body)
    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_image.png")  # type: ignore


def test_get_workflow_basic(api_client):
    body = ImageRequest(
        prompt="A beautiful mountain landscape",
        model=model,
        max_width=512,
        max_height=512,
        comfy_workflow=json.load(open("../assets/workflows/text2Image.json", encoding="utf-8")),
    )
    image_id = create_image(api_client, body)

    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_workflow_basic.png")  # type: ignore


def test_get_workflow_advanced(api_client):
    body = ImageRequest(
        prompt="tornado on farm feild, enhance keep original elements, Detailed, 8k, DSLR photo, photorealistic",
        model=model,
        max_width=512,
        max_height=512,
        image=image_to_base64("../assets/color_v001.jpeg"),
        mask=image_to_base64("../assets/mask_v001.png"),
        comfy_workflow=json.load(open("../assets/workflows/image2Image.json", encoding="utf-8")),
    )
    image_id = create_image(api_client, body)

    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_workflow_advanced.png")  # type: ignore

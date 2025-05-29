import json
import os
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import images_create, images_get
from generated.api_client.client import AuthenticatedClient, Client
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse

model = ImageRequestModel("sd1.5")


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DEF_DIF_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DEF_DIF_API_KEY", ""),
    )


def create_image(api_client, comfy_workflow=None):
    """Helper function to create an image and return its ID."""
    request = ImageRequest(
        prompt="A beautiful mountain landscape",
        model=model,
        max_width=512,
        max_height=512,
        comfy_workflow=comfy_workflow,
    )

    response = images_create.sync_detailed(client=api_client, body=request)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


def test_create_image(api_client):
    """Test creating an image through the API."""
    image_id = create_image(api_client)
    assert isinstance(image_id, UUID)


def test_get_image(api_client):
    """Test retrieving an image by ID."""
    image_id = create_image(api_client)

    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"


def test_get_workflow_basic(api_client):
    """Test retrieving an image workflow by ID."""
    comfy_workflow = json.load(open("../assets/workflow_basic_v001.json", encoding="utf-8"))
    image_id = create_image(api_client, comfy_workflow=comfy_workflow)
    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"  # Should be pending since we didn't wait


def test_get_workflow_advanced(api_client):
    """Test retrieving an image workflow by ID."""
    comfy_workflow = json.load(open("../assets/workflow_basic_v001.json", encoding="utf-8"))
    image_id = create_image(api_client, comfy_workflow=comfy_workflow)
    response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"  # Should be pending since we didn't wait

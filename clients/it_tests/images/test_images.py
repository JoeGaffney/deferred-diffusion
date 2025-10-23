import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import images_create, images_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_request_model import ImageRequestModel
from generated.api_client.models.image_response import ImageResponse
from utils import image_to_base64, save_image_and_assert_file_exists

model = ImageRequestModel("sd-xl")
output_dir = "../tmp/output/it-tests/images"
image_a = image_to_base64("../../assets/color_v001.jpeg")
image_b = image_to_base64("../../assets/style_v001.jpeg")
video_a = image_to_base64("../../assets/video_v001.mp4")


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
        raise_on_unexpected_status=True,
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
    body = ImageRequest(prompt="A beautiful mountain landscape", model=model, width=512, height=512)
    image_id = create_image(api_client, body)
    assert isinstance(image_id, UUID)


def test_get_image(api_client):
    body = ImageRequest(prompt="A beautiful mountain landscape", model=model, width=512, height=512)
    image_id = create_image(api_client, body)

    for _ in range(20):  # Retry up to 20 times
        time.sleep(5)
        response = images_get.sync_detailed(id=image_id, client=api_client)
        if isinstance(response.parsed, ImageResponse) and response.parsed.status in ["SUCCESS", "COMPLETED"]:
            break

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_image.png")  # type: ignore


def test_multi_submit_image(api_client):
    "test that multiple image requests can be submitted successfully"
    body = ImageRequest(prompt="A beautiful mountain landscape", image=image_a, model=model, width=512, height=512)

    for _ in range(3):
        image_id = create_image(api_client, body)
        assert isinstance(image_id, UUID)


def test_multi_get_image(api_client):
    "test that multiple image requests can be retrieved successfully"
    body = ImageRequest(prompt="A beautiful mountain landscape", model=model, width=512, height=512)
    image_id = create_image(api_client, body)

    for _ in range(10):
        response = images_get.sync_detailed(id=image_id, client=api_client)
        assert response.status_code == HTTPStatus.OK

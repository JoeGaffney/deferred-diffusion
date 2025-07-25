import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.texts import texts_create, texts_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models.message_content import MessageContent
from generated.api_client.models.message_item import MessageItem
from generated.api_client.models.text_create_response import TextCreateResponse
from generated.api_client.models.text_request import TextRequest
from generated.api_client.models.text_request_model import TextRequestModel
from generated.api_client.models.text_response import TextResponse
from utils import image_to_base64

model = TextRequestModel("qwen-2-5")
model = TextRequestModel("external-gpt-4")
# model = TextRequestModel("external-gpt-4-1")

image_a = image_to_base64("../assets/color_v001.jpeg")
image_b = image_to_base64("../assets/style_v001.jpeg")
video_a = image_to_base64("../assets/video_v001.mp4")


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
    )


def create_text(api_client):
    """Helper function to create an image and return its ID."""
    request = TextRequest(
        model=model,
        messages=[
            MessageItem(
                role="user",
                content=[
                    MessageContent(
                        type_="input_text",
                        text="Generate a prompt for SD image generation to generate similar images.",
                    ),
                ],
            ),
        ],
        images=[image_a],
    )

    response = texts_create.sync_detailed(client=api_client, body=request)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, TextCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


def test_create_text(api_client):
    """Test creating an image through the API."""
    image_id = create_text(api_client)
    assert isinstance(image_id, UUID)


def test_get_text(api_client):
    """Test retrieving an image by ID."""
    image_id = create_text(api_client)
    time.sleep(10)  # Wait for the task to be processed

    response = texts_get.sync_detailed(id=image_id, client=api_client)
    print(response.parsed)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, TextResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"

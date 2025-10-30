import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import images_create_local, images_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models.image_create_response import ImageCreateResponse
from generated.api_client.models.image_request import ImageRequest
from generated.api_client.models.image_response import ImageResponse
from generated.api_client.models.images_create_local_model import ImagesCreateLocalModel

model = ImagesCreateLocalModel("sd-xl")


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
        raise_on_unexpected_status=True,
    )


def create_image(api_client, body: ImageRequest) -> UUID:
    """Helper function to create an image and return its ID."""

    response = images_create_local.sync_detailed(client=api_client, model=model, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


def test_connection_reset_edge(api_client):
    """Poll repeatedly to hit the server keep-alive edge case"""
    # Create a new image request
    body = ImageRequest(prompt="Test polling edge", width=512, height=512)
    response = images_create_local.sync(client=api_client, model=model, body=body)
    assert isinstance(response, ImageCreateResponse)
    assert isinstance(response.id, UUID)

    # Poll exactly at the server's keep-alive timeout
    poll_interval = 5  # match Uvicorn default timeout_keep_alive
    print("Polling at 5s interval to trigger potential reset...")

    for i in range(10):
        time.sleep(poll_interval)
        result = images_get.sync(id=response.id, client=api_client)
        assert isinstance(result, ImageResponse)
        print(f"Poll {i+1}: status={result.status}")
        # try:
        #     result = images_get.sync(id=response.id, client=api_client)
        #     print(f"Poll {i+1}: status={result.status}")
        # except Exception as e:
        #     # This should happen on unlucky iteration
        #     print(f"Poll {i+1}: caught exception! {type(e).__name__}: {e}")
        #     break

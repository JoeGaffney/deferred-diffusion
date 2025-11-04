import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import images_create_local, images_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models import (
    ImageCreateResponse,
    ImageRequest,
    ImageResponse,
    ImagesCreateLocalModel,
)
from utils import image_a, save_image_and_assert_file_exists

models = [ImagesCreateLocalModel("sd-xl")]


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
        raise_on_unexpected_status=True,
    )


def create_image(api_client, model: ImagesCreateLocalModel, body: ImageRequest) -> UUID:
    response = images_create_local.sync_detailed(client=api_client, model=model, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


@pytest.mark.local
@pytest.mark.parametrize("model", models)
def test_create_image(api_client, model):
    body = ImageRequest(prompt="A beautiful mountain landscape", width=512, height=512)
    image_id = create_image(api_client, model, body)

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
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"test_images_{model}.png")  # type: ignore


@pytest.mark.local
@pytest.mark.parametrize("model", models)
def test_multi_submit_image(api_client, model):
    body = ImageRequest(prompt="A beautiful mountain landscape", image=image_a, width=512, height=512)

    for _ in range(3):
        image_id = create_image(api_client, model, body)
        assert isinstance(image_id, UUID)


@pytest.mark.local
@pytest.mark.parametrize("model", models)
def test_multi_get_image(api_client, model):
    body = ImageRequest(prompt="A beautiful mountain landscape", width=512, height=512)
    image_id = create_image(api_client, model, body)

    for _ in range(10):
        response = images_get.sync_detailed(id=image_id, client=api_client)
        assert response.status_code == HTTPStatus.OK


@pytest.mark.local
@pytest.mark.parametrize("model", models)
def test_connection_reset_edge(api_client, model):
    body = ImageRequest(prompt="Test polling edge", width=512, height=512)
    response = images_create_local.sync(client=api_client, model=model, body=body)

    assert isinstance(response, ImageCreateResponse)
    assert isinstance(response.id, UUID)

    # Poll exactly at the server's keep-alive timeout
    poll_interval = 5  # match Uvicorn default timeout_keep_alive
    print("Polling at 5s interval to trigger potential reset...")

    for i in range(5):
        time.sleep(poll_interval)
        result = images_get.sync(id=response.id, client=api_client)
        assert isinstance(result, ImageResponse)
        print(f"Poll {i+1}: status={result.status}")

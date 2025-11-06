import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.images import images_create, images_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models import (
    ImageCreateResponse,
    ImageRequest,
    ImageRequestModel,
    ImageResponse,
)
from utils import image_to_base64, save_image_and_assert_file_exists

models = [ImageRequestModel("flux-1-pro")]


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
        raise_on_unexpected_status=True,
    )


def create_image(api_client, body: ImageRequest) -> UUID:
    response = images_create.sync_detailed(client=api_client, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ImageCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


@pytest.mark.external
@pytest.mark.parametrize("model", models)
def test_create_image(api_client, model):
    body = ImageRequest(model=model, prompt="A beautiful mountain landscape", width=512, height=512)
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
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"test_images_{model}.png")  # type: ignore

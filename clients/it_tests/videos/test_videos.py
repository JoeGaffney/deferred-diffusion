import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.videos import videos_create_local, videos_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models import (
    VideoCreateResponse,
    VideoRequest,
    VideoResponse,
    VideosCreateLocalModel,
)
from utils import image_to_base64, save_image_and_assert_file_exists

output_dir = "../tmp/output/it-tests/videos"


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
    )


def create_video(api_client, model: VideosCreateLocalModel, body: VideoRequest) -> UUID:
    """Helper function to create an image and return its ID."""

    response = videos_create_local.sync_detailed(client=api_client, model=model, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, VideoCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


def test_get_ltx(api_client):
    body = VideoRequest(
        image=image_to_base64("../../assets/color_v002.png"),
        prompt="A man with short gray hair plays a red electric guitar.",
        num_frames=24,
    )
    image_id = create_video(api_client, VideosCreateLocalModel("ltx-video"), body)

    for _ in range(20):  # Retry up to 20 times
        time.sleep(10)
        response = videos_get.sync_detailed(id=image_id, client=api_client)
        if isinstance(response.parsed, VideoResponse) and response.parsed.status in ["SUCCESS", "COMPLETED"]:
            break

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, VideoResponse)
    assert response.parsed.id == image_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_ltx.mp4")  # type: ignore

import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.videos import videos_create, videos_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models import (
    VideoCreateResponse,
    VideoRequest,
    VideoRequestModel,
    VideoResponse,
)
from utils import assert_logs_exist, image_c, save_image_and_assert_file_exists

models = [VideoRequestModel("runway-gen-4")]


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
    )


def create_video(api_client, body: VideoRequest) -> UUID:
    """Helper function to create a video and return its ID."""

    response = videos_create.sync_detailed(client=api_client, body=body)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, VideoCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


@pytest.mark.external
@pytest.mark.parametrize("model", models)
def test_create_video(api_client, model):
    body = VideoRequest(
        model=model,
        image=image_c,
        prompt="A man with short gray hair plays a red electric guitar.",
        num_frames=24,
    )
    video_id = create_video(api_client, body)

    for _ in range(20):  # Retry up to 20 times
        time.sleep(10)
        response = videos_get.sync_detailed(id=video_id, client=api_client)
        if isinstance(response.parsed, VideoResponse) and response.parsed.status in ["SUCCESS", "COMPLETED"]:
            break

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, VideoResponse)
    assert response.parsed.id == video_id
    assert response.parsed.status == "SUCCESS"
    save_image_and_assert_file_exists(response.parsed.result.base64_data, f"test_videos_{model}.mp4")  # type: ignore
    assert_logs_exist(response.parsed.logs)

import json
import os
from http import HTTPStatus

import pytest

from generated.api_client.api.videos import videos_get_workflow_schema
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models.comfy_workflow_response import ComfyWorkflowResponse
from generated.api_client.models.video_request_model import VideoRequestModel

model = VideoRequestModel("LTX-Video")
output_dir = "../tmp/output/it-tests/videos"


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DEF_DIF_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DEF_DIF_API_KEY", ""),
    )


def test_get_workflow_schema(api_client):
    """Test to ensure the workflow schema can be retrieved."""
    response = videos_get_workflow_schema.sync_detailed(client=api_client)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, ComfyWorkflowResponse)

    # Save the dictionary to a JSON file
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/workflow_schema.json", "w", encoding="utf-8") as f:
        json.dump(response.parsed.to_dict(), f, indent=4)


# def test_get_image(api_client):
#     body = ImageRequest(prompt="A beautiful mountain landscape", model=model, max_width=512, max_height=512)
#     image_id = create_image(api_client, body)
#     response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

#     assert response.status_code == HTTPStatus.OK
#     assert response.parsed is not None
#     assert isinstance(response.parsed, ImageResponse)
#     assert response.parsed.id == image_id
#     assert response.parsed.status == "SUCCESS"
#     save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_image.png")  # type: ignore


# def test_get_workflow_basic(api_client):
#     body = ImageRequest(
#         prompt="A beautiful mountain landscape",
#         model=model,
#         max_width=512,
#         max_height=512,
#         comfy_workflow=json.load(open("../assets/workflows/text2Image.json", encoding="utf-8")),
#     )
#     image_id = create_image(api_client, body)

#     response = images_get.sync_detailed(id=image_id, client=api_client, wait=True)

#     assert response.status_code == HTTPStatus.OK
#     assert response.parsed is not None
#     assert isinstance(response.parsed, ImageResponse)
#     assert response.parsed.id == image_id
#     assert response.parsed.status == "SUCCESS"
#     save_image_and_assert_file_exists(response.parsed.result.base64_data, f"{output_dir}/test_get_workflow_basic.png")  # type: ignore

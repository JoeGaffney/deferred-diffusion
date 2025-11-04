import os
import time
from http import HTTPStatus
from uuid import UUID

import pytest

from generated.api_client.api.texts import texts_create_external, texts_get
from generated.api_client.client import AuthenticatedClient
from generated.api_client.models import (
    MessageContent,
    MessageItem,
    TextCreateResponse,
    TextRequest,
    TextResponse,
    TextsCreateExternalModel,
)
from utils import image_a

external_models = [TextsCreateExternalModel("gpt-4")]


@pytest.fixture
def api_client():
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
    )


def create_text(api_client, model):
    """Helper function to create a text and return its ID."""
    request = TextRequest(
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

    response = texts_create_external.sync_detailed(client=api_client, model=model, body=request)

    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, TextCreateResponse)
    assert isinstance(response.parsed.id, UUID)
    assert response.parsed.status == "PENDING"

    return response.parsed.id


@pytest.mark.external
@pytest.mark.parametrize("model", external_models)
def test_create_text(api_client, model):
    """Test retrieving a text by ID."""
    text_id = create_text(api_client, model)

    for _ in range(20):  # Retry up to 20 times
        time.sleep(5)
        response = texts_get.sync_detailed(id=text_id, client=api_client)
        if isinstance(response.parsed, TextResponse) and response.parsed.status in ["SUCCESS", "COMPLETED"]:
            break

    print(response.parsed)
    assert response.status_code == HTTPStatus.OK
    assert response.parsed is not None
    assert isinstance(response.parsed, TextResponse)
    assert response.parsed.id == text_id
    assert response.parsed.status == "SUCCESS"

import os

from generated.api_client.client import AuthenticatedClient

MAX_ADDITIONAL_IMAGES = 3


client = AuthenticatedClient(
    base_url=os.getenv("DEF_DIF_API_ADDRESS", "http://127.0.0.1:5000"),
    token=os.getenv("DEF_DIF_API_KEY", ""),
    raise_on_unexpected_status=True,
)

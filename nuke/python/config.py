import os

from generated.api_client.client import AuthenticatedClient

MAX_ADDITIONAL_IMAGES = 3


client = AuthenticatedClient(
    base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
    token=os.getenv("DDIFFUSION_API_KEY", ""),
)

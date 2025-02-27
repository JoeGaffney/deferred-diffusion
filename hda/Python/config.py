import os

from api_client.api_client import Client

MAX_ADDITIONAL_IMAGES = 3

base_url = os.getenv("DD_SERVER_ADDRESS", "http://127.0.0.1:5000")

client = Client(base_url=base_url)

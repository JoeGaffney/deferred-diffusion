import os

from api_client.api_client import Client

base_url = os.getenv("DD_SERVER_ADDRESS", "http://127.0.0.1:5000")

client = Client(base_url=base_url)

from api_client.api_client.models import TextRequest, TextResponse
from api_client.api_client.api.text import create_api_text_post
from api_client.api_client.types import Response

def main(node=None):
    body_dict = {
        "prompt": "Detailed, 8k, photorealistic",
        "messages": [],
    }
    body = TextRequest(prompt=body_dict["prompt"], messages=body_dict["messages"])

    # my_data: TextResponse = create_api_text_post.sync(client=client,body=body) 
    # or if you need more info (e.g. status_code)
    response: Response[TextResponse] = create_api_text_post.sync_detailed(client=client,body=body)
    print(response)
    print(response.parsed)

    return 
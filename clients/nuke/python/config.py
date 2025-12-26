import getpass
import os
import socket

import httpx

from generated.api_client.client import AuthenticatedClient

client = AuthenticatedClient(
    base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
    token=os.getenv("DDIFFUSION_API_KEY", ""),
    raise_on_unexpected_status=True,
    timeout=httpx.Timeout(180),  # timeout in seconds
    headers={
        "x-user-id": getpass.getuser(),
        "x-machine-id": socket.gethostname(),
    },
)

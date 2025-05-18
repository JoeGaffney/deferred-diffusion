import os

from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="Authorization")


def verify_token(
    authorization: str = Depends(api_key_header),
) -> None:
    if not authorization:
        raise HTTPException(status_code=403, detail="Missing authorization token")

    # Get API key from environment variable
    api_keys_env = os.environ.get("DEF_DIF_API_KEYS")
    if not api_keys_env:
        raise HTTPException(status_code=403, detail="Server configuration error: authorization not set")

    api_keys = api_keys_env.split(",")

    # Check if token matches expected format and is in the list of valid keys
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid token format")

    token = authorization.replace("Bearer ", "")
    if token not in api_keys:
        raise HTTPException(status_code=403, detail="Invalid or expired token")

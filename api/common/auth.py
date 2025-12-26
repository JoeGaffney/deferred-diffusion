import os

import redis
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from slowapi.util import get_ipaddr, get_remote_address

from common.limiter import get_rate_limit_key

api_key_header = APIKeyHeader(name="Authorization")


# Connect to Redis
redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
r = redis.from_url(redis_url, decode_responses=True)
_KEY_PREFIX = "DDIFFUSION_API_KEY"


def verify_token(
    authorization: str = Depends(api_key_header),
) -> None:
    if not authorization:
        raise HTTPException(status_code=403, detail="Missing authorization token")

    token = authorization.replace("Bearer ", "")

    # Check Redis for the key
    key_name = f"{_KEY_PREFIX}:{token}"

    # Check if key exists and is active
    # hget returns None if key or field doesn't exist
    if r.hget(key_name, "active") != "1":
        raise HTTPException(status_code=403, detail="Invalid or revoked token")


def request_identity(request: Request) -> dict:
    """
    Extracts identity information from the request to pass to the worker.
    """

    return {
        "user_id": request.headers.get("x-user-id", "unknown"),
        "machine_id": request.headers.get("x-machine-id", "unknown"),
        "client_ip": get_remote_address(request),
        "hash_key": get_rate_limit_key(request),
    }

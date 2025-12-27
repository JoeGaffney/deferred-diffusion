import os

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from common.api_key_manager import key_manager
from common.logger import log_request
from common.schemas import Identity


def get_remote_address(request: Request) -> str:
    """
    Returns the ip address for the current request (or 127.0.0.1 if none found)
    """
    if not request.client or not request.client.host:
        return "127.0.0.1"

    return request.client.host


api_key_header = APIKeyHeader(name="Authorization")
admin_api_key_header = APIKeyHeader(name="Authorization")


async def verify_token(
    request: Request,
    authorization: str = Depends(api_key_header),
) -> Identity:
    if not authorization:
        raise HTTPException(status_code=403, detail="Missing authorization token")

    token = authorization.replace("Bearer ", "")

    if not key_manager.is_active(token):
        raise HTTPException(status_code=403, detail="Invalid or revoked token")

    key_hash = key_manager.hash_token(token)
    key_name = key_manager.get_name(key_hash)

    identity = Identity(
        user_id=request.headers.get("x-user-id", "unknown"),
        machine_id=request.headers.get("x-machine-id", "unknown"),
        client_ip=get_remote_address(request),
        key_name=key_name,
        key_hash=key_hash,
    )
    await log_request(request, identity)

    # Only rate limit POST requests (task creation) so polling doesn't consume quota
    if request.method == "POST":
        limit = int(os.getenv("CREATES_PER_MINUTE", "30"))

        if not key_manager.check_rate_limit(key_hash, limit=limit, window=60):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    return identity


# Restrict all admin endpoints to those with the Admin Key
def admin_only(authorization: str = Depends(admin_api_key_header)):
    expected_key = os.getenv("DDIFFUSION_ADMIN_KEY", "supersecretadminkey")
    token = authorization.replace("Bearer ", "")

    if token != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden - Invalid Admin Key")

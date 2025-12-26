import os

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from slowapi.util import get_remote_address

from common.api_key_manager import key_manager
from common.limiter import get_rate_limit_key

api_key_header = APIKeyHeader(name="Authorization")
admin_api_key_header = APIKeyHeader(name="Authorization")


def verify_token(
    authorization: str = Depends(api_key_header),
) -> None:
    if not authorization:
        raise HTTPException(status_code=403, detail="Missing authorization token")

    token = authorization.replace("Bearer ", "")

    if not key_manager.is_active(token):
        raise HTTPException(status_code=403, detail="Invalid or revoked token")


def request_identity(request: Request) -> dict:
    """
    Extracts identity information from the request to pass to the worker.
    """
    auth_header = request.headers.get("Authorization")
    key_name = key_manager.get_name(auth_header.replace("Bearer ", ""))

    return {
        "user_id": request.headers.get("x-user-id", "unknown"),
        "machine_id": request.headers.get("x-machine-id", "unknown"),
        "client_ip": get_remote_address(request),
        "key_name": key_name,
        "key_hash": get_rate_limit_key(request),
    }


# Restrict all admin endpoints to those with the Admin Key
def admin_only(authorization: str = Depends(admin_api_key_header)):
    expected_key = os.getenv("DDIFFUSION_ADMIN_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Admin key not set")

    token = authorization.replace("Bearer ", "")

    if token != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden - Invalid Admin Key")

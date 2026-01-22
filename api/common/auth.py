import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader

from common.config import settings
from common.logger import log_request
from common.redis_manager import redis_manager
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

    key_data = redis_manager.verify_token(token)
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid or revoked token")

    identity = Identity(
        user_id=request.headers.get("x-user-id", "unknown"),
        machine_id=request.headers.get("x-machine-id", "unknown"),
        client_ip=get_remote_address(request),
        key_name=key_data.name,
        key_id=key_data.key_id,
    )
    await log_request(request, identity)

    # Only limit POST requests (task creation)
    if request.method == "POST":
        waiting_tasks = redis_manager.waiting_tasks()
        if waiting_tasks >= settings.task_backlog_limit:
            raise HTTPException(
                status_code=429,
                detail=f"Too many waiting tasks {waiting_tasks} / {settings.task_backlog_limit}, please try later",
            )

    return identity


# Restrict all admin endpoints to those with the Admin Key
def admin_only(authorization: str = Depends(admin_api_key_header)):
    token = authorization.replace("Bearer ", "")

    if not secrets.compare_digest(token, settings.ddiffusion_admin_key):
        raise HTTPException(status_code=403, detail="Forbidden")

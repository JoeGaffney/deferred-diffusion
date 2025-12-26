import datetime
import os
import secrets

import redis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader

redis_url = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
r = redis.from_url(redis_url, decode_responses=True)
_KEY_PREFIX = "DDIFFUSION_API_KEY"

admin_api_key_header = APIKeyHeader(name="Authorization")


# Restrict all admin endpoints to those with the Admin Key
def admin_only(authorization: str = Depends(admin_api_key_header)):
    expected_key = os.getenv("DDIFFUSION_ADMIN_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Admin key not set")

    token = authorization.replace("Bearer ", "")

    if token != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden - Invalid Admin Key")


router = APIRouter(
    prefix="/admin", tags=["Admin"], dependencies=[Depends(admin_only)]  # This will apply to all routes
)


@router.post("/keys", operation_id="keys_create")
def create(name: str):

    def create_api_key(name: str) -> str:
        token = secrets.token_urlsafe(32)
        key_name = f"{_KEY_PREFIX}:{token}"
        r.hset(
            key_name,
            mapping={
                "name": name,
                "active": "1",
                "created_at": datetime.datetime.utcnow().isoformat(),
            },
        )
        return token

    token = create_api_key(name)
    return {"api_key": token, "name": name}


@router.get("/keys", operation_id="keys_list")
def list():
    def list_api_keys():
        keys = []
        for key in r.scan_iter(f"{_KEY_PREFIX}:*"):
            data = r.hgetall(key)
            keys.append({"api_key": key.split(f"{_KEY_PREFIX}:")[1], **data})  # Type: ignore

        return keys

    return list_api_keys()


@router.delete("/keys", operation_id="keys_delete")
def delete(token: str):
    def revoke_api_key(token: str):
        key_name = f"{_KEY_PREFIX}:{token}"
        if r.exists(key_name):
            r.hset(key_name, "active", "0")
            return True
        return False

    if revoke_api_key(token):
        return {"revoked": True}

    raise HTTPException(404, "Key not found")

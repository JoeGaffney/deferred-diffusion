from fastapi import APIRouter, Depends, HTTPException, Query

from common.auth import admin_only
from common.redis_manager import redis_manager

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(admin_only)])


@router.post("/keys", operation_id="keys_create")
def create(name: str = Query(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9 _-]+$")):
    try:
        token = redis_manager.create_key(name)
        return {"api_key": token, "name": name}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/keys", operation_id="keys_list")
def list():
    return redis_manager.list_keys()


@router.delete("/keys/{key_id}", operation_id="keys_delete")
def delete(key_id: str):
    if redis_manager.delete_key(key_id):
        return {"deleted": True}

    raise HTTPException(404, "Key not found")

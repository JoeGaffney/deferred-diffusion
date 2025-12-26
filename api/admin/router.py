from fastapi import APIRouter, Depends, HTTPException

from common.api_key_manager import key_manager
from common.auth import admin_only

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(admin_only)])


@router.post("/keys", operation_id="keys_create")
def create(name: str):
    token = key_manager.create_key(name)
    return {"api_key": token, "name": name}


@router.get("/keys", operation_id="keys_list")
def list():
    return key_manager.list_keys()


@router.delete("/keys", operation_id="keys_delete")
def delete(token: str):
    if key_manager.revoke_key(token):
        return {"revoked": True}

    raise HTTPException(404, "Key not found")

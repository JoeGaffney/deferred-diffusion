import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from common.config import settings
from common.storage import verify_signed_url

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id:path}", operation_id="files_get")
async def get(file_id: str, expires: int, sig: str):
    # Verify the signature against the full path
    if not verify_signed_url(f"/api/files/{file_id}", "GET", expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    storage_root = os.path.abspath(settings.storage_dir)
    full_path = os.path.abspath(os.path.join(storage_root, file_id))

    # Ensure the resolved path is within the storage root to prevent path traversal
    if os.path.commonpath([storage_root, full_path]) != storage_root:
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"File not found: {full_path}")

    return FileResponse(full_path)

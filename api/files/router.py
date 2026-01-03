from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from common.config import settings
from common.storage import verify_signed_url

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id:path}", operation_id="files_get")
async def get(file_id: str, expires: int, sig: str):
    if not verify_signed_url(file_id, "GET", expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Reject any absolute paths / traversal attempts from user input immediately
    user_path = Path(file_id)
    if ".." in user_path.parts or user_path.is_absolute():
        raise HTTPException(status_code=403, detail="Invalid file path")

    # Normalize path and prevent path traversal
    storage_root = Path(settings.storage_dir).resolve()
    full_path = (storage_root / user_path).resolve()

    if not full_path.is_relative_to(storage_root):
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    return FileResponse(str(full_path))

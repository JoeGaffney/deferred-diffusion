from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from common.config import settings
from common.storage import verify_signed_url

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id:path}", operation_id="files_get")
async def get(file_id: str, expires: int, sig: str):
    # 1. Cryptographic validation (The real security)
    if not verify_signed_url(file_id, "GET", expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # 2. Explicit sanitization (To satisfy scanners)
    if ".." in file_id or file_id.startswith("/"):
        raise HTTPException(status_code=403, detail="Invalid file path")

    storage_root = Path(settings.storage_dir).resolve()

    try:
        # 3. Path resolution
        full_path = (storage_root / file_id).resolve()
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    # 4. Boundary check
    if not full_path.is_relative_to(storage_root):
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_id}")

    return FileResponse(str(full_path))

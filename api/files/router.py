from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from common.config import settings
from common.storage import verify_signed_url

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id:path}", operation_id="files_get")
async def get(file_id: str, expires: int, sig: str):
    # Verify the signature against the full path
    if not verify_signed_url(file_id, "GET", expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Use pathlib for more robust path handling and to satisfy security scanners.
    # .resolve() ensures we have an absolute, normalized path.
    storage_root = Path(settings.storage_dir).resolve()

    # Ensure the file_id is treated as a relative path to the storage root.
    # lstrip("/") prevents it from being treated as an absolute path if it starts with /.
    # We then resolve it to handle any '..' segments.
    try:
        full_path = (storage_root / file_id.lstrip("/")).resolve()
    except Exception:
        raise HTTPException(status_code=403, detail="Invalid file path")

    # Ensure the resolved path is still within the storage root to prevent path traversal.
    if not full_path.is_relative_to(storage_root):
        raise HTTPException(status_code=403, detail="Invalid file path")

    if not full_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_id}")

    if not full_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {file_id}")

    return FileResponse(full_path)

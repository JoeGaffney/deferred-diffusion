import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from common.config import settings
from common.storage import verify_signed_url

router = APIRouter(prefix="/files", tags=["files"])


@router.get("/{file_id}", operation_id="files_get")
async def download_file(file_id: str, expires: int, sig: str):
    """
    DOWNLOAD ROUTE: Requires a signature.
    This is what the MCP/Agent returns to the user.
    """
    # Verify the signature against the full path
    if not verify_signed_url(f"/api/files/{file_id}", "GET", expires, sig):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    file_path = os.path.join(settings.storage_dir, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)

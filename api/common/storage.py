import base64
import hashlib
import hmac
import os
import time
from typing import Optional, Union
from uuid import UUID

from common.config import settings
from common.logger import logger


def generate_signed_url(base_url: str, path: str, method: str = "GET", expires_in: int = 3600) -> str:
    """
    Generates a signed URL for internal studio use.
    path: e.g. "/api/files/task_123"
    """
    expires = int(time.time()) + expires_in
    # We sign the method, path, and expiration to prevent tampering
    msg = f"{method}:{path}:{expires}"
    sig = hmac.new(settings.encoded_storage_key, msg.encode(), hashlib.sha256).hexdigest()

    sep = "&" if "?" in path else "?"
    return f"{base_url}{path}{sep}expires={expires}&sig={sig}"


def verify_signed_url(path: str, method: str, expires: int, sig: str) -> bool:
    """
    Verifies a signed URL.
    """
    if expires < time.time():
        return False

    msg = f"{method}:{path}:{expires}"
    expected = hmac.new(settings.encoded_storage_key, msg.encode(), hashlib.sha256).hexdigest()

    return hmac.compare_digest(sig, expected)


def promote_result_to_storage(
    task_id: Union[UUID, str],
    base64_data: Optional[Union[str, bytes]],
    extension: str,
    base_url: Optional[str] = None,
) -> Optional[str]:
    """
    Lazy caches a base64 result from Redis to disk and returns a signed download URL.
    """
    if not base64_data:
        return None

    file_id = f"{task_id}.{extension}"
    file_path = os.path.join(settings.storage_dir, file_id)

    if not os.path.exists(file_path):
        os.makedirs(settings.storage_dir, exist_ok=True)
        try:
            # Ensure we are dealing with bytes
            if isinstance(base64_data, str):
                data_bytes = base64.b64decode(base64_data)
            else:
                data_bytes = base64_data

            with open(file_path, "wb") as f:
                f.write(data_bytes)
            logger.info(f"Promoted result {task_id} to storage: {file_path}")
        except Exception as e:
            logger.error(f"Failed to promote result {task_id} to storage: {e}")
            return None

    return generate_signed_url(base_url or settings.base_url, f"/api/files/{file_id}", method="GET")

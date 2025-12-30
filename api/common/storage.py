import base64
import hashlib
import hmac
import os
import time
from uuid import UUID

from pydantic import HttpUrl

from common.config import settings
from common.logger import logger


def generate_signed_url(base_url: str, path: str, method: str = "GET", expires_in: int = 3600) -> HttpUrl:
    """
    Generates a signed URL for internal studio use.
    path: e.g. "/api/files/task_123"
    """
    expires = int(time.time()) + expires_in
    # We sign the method, path, and expiration to prevent tampering
    msg = f"{method}:{path}:{expires}"
    sig = hmac.new(settings.encoded_storage_key, msg.encode(), hashlib.sha256).hexdigest()

    sep = "&" if "?" in path else "?"
    result = HttpUrl(f"{base_url}{path}{sep}expires={expires}&sig={sig}")
    logger.info(f"Generated signed URL: {result}")
    return result


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
    task_id: UUID,
    base64_data: bytes,  # The raw binary data, not base64-encoded string, pydantic decodes allready
    extension: str,
    base_url: str,
    index: int = 0,
) -> HttpUrl:
    """
    Lazy caches a base64 result from Redis to disk and returns a signed download URL.
    """

    file_id = f"{task_id}_{index}.{extension}"
    file_path = os.path.join(settings.storage_dir, file_id)

    if not os.path.exists(file_path):
        os.makedirs(settings.storage_dir, exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                f.write(base64_data)
            logger.info(f"Promoted result {task_id} to storage: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to promote result to storage: {e}")

    return generate_signed_url(base_url, f"/api/files/{file_id}", method="GET")

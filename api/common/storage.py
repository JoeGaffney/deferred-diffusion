import hashlib
import hmac
import os
import time
from pathlib import Path
from uuid import UUID

from pydantic import HttpUrl

from common.config import settings
from common.logger import logger


def _get_signature(file_id: str, method: str, expires: int) -> str:
    """
    Generates HMAC signature for a given file ID, method, and expiration timestamp.
    """
    msg = f"{method}:storage:{file_id}:{expires}"
    signature = hmac.new(settings.encoded_storage_key, msg.encode(), hashlib.sha256).hexdigest()
    return signature


def generate_signed_url(file_id: str, method: str = "GET", expires_in: int = 3600) -> HttpUrl:
    """
    Generates a signed URL for internal studio use.
    file_id: e.g. "flux-1/423e6f8c-5b6d-4d3a-9f7e-2c3d4e5f6a7b_0.png"
    """

    expires = int(time.time()) + expires_in
    signature = _get_signature(file_id, method, expires)

    base_url = f"{settings.ddiffusion_storage_address}/api/files/"
    result = HttpUrl(f"{base_url}{file_id}?expires={expires}&sig={signature}")

    logger.info(f"Generated signed URL: {result}")
    return result


def verify_signed_url(file_id: str, method: str, expires: int, sig: str) -> bool:
    """
    Verifies a signed URL.
    """
    if expires < time.time():
        return False

    expected = _get_signature(file_id, method, expires)
    return hmac.compare_digest(sig, expected)


def signed_url_for_file(file_id: str) -> HttpUrl:
    """
    Generates a signed URL for a given file ID.
    """
    full_path = Path(settings.storage_dir) / Path(file_id)
    if full_path.exists() is False:
        raise FileNotFoundError(f"File not found for signed URL generation: {full_path}")

    return generate_signed_url(file_id, method="GET", expires_in=settings.signed_url_expiry_seconds)


def promote_result_to_storage(
    task_id: UUID,
    base64_data: bytes,  # The raw binary data, not base64-encoded string, pydantic decodes allready
    extension: str,
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

    return generate_signed_url(file_id, method="GET", expires_in=settings.signed_url_expiry_seconds)

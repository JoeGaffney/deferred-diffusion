import hashlib
import hmac
import time
from pathlib import Path

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

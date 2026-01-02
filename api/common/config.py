import hashlib
import logging
import os
import tempfile
import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=True)

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    ddiffusion_admin_key: str = "supersecretadminkey"
    ddiffusion_storage_address: str = (
        "http://127.0.0.1:5000"  # external services should use this to reach the API / used for signed URLs
    )
    ddiffusion_storage_directory: str = os.path.join(tempfile.gettempdir(), "deferred-diffusion", "storage")
    signed_url_expiry_seconds: int = 3600  # 1 hour
    creates_per_minute: int = 30
    enable_mcp: bool = True

    @property
    def encoded_storage_key(self) -> bytes:
        # Derive a unique signing key from the admin key to avoid using it directly.
        # This satisfies security recommendations while still being based on the admin key.
        return hashlib.sha256(f"{self.ddiffusion_admin_key}:internal-storage-signing-v1".encode()).digest()

    @property
    def storage_dir(self) -> str:
        subdir = self.ddiffusion_storage_directory
        os.makedirs(subdir, exist_ok=True)
        return subdir

    @field_validator("ddiffusion_admin_key")
    @classmethod
    def validate_admin_key(cls, v: str) -> str:
        if v == "supersecretadminkey":
            warnings.warn(
                "Security risk: Using default value for DDIFFUSION_ADMIN_KEY. "
                "Change this in production via environment variables.",
                UserWarning,
            )
        return v


settings = Settings()  # type: ignore

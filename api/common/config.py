import hashlib
import logging
import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=True)

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    ddiffusion_admin_key: str = Field(min_length=32)
    # external services should use this to reach the API / used for signed URLs
    ddiffusion_storage_address: str = "http://127.0.0.1:5000"
    ddiffusion_storage_directory: str = "/STORAGE"
    flower_url: str = "http://flower:5555"
    signed_url_expiry_seconds: int = 3600 * 1  # 1 hour
    task_backlog_limit: int = 100  # Max number of waiting tasks allowed before rejecting new ones
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


settings = Settings()  # type: ignore

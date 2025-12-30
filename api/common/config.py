import logging
import os
import tempfile
import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def get_tmp_dir() -> str:
    subdir = os.path.join(tempfile.gettempdir(), "deferred-diffusion", "api")
    os.makedirs(subdir, exist_ok=True)
    return subdir


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=True)

    # Redis / Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Security
    ddiffusion_admin_key: str = "supersecretadminkey"

    # Storage
    storage_dir: str = get_tmp_dir()
    base_url: str = "http://localhost:5000"

    # Rate limiting
    creates_per_minute: int = 30

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

    @property
    def encoded_storage_key(self) -> bytes:
        extra = f"{self.ddiffusion_admin_key}:internal-storage-signing-v1"
        return extra.encode()


settings = Settings()  # type: ignore

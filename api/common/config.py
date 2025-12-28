import logging
import warnings

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=True)

    # Redis / Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"

    # Security
    ddiffusion_admin_key: str = "supersecretadminkey"

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


settings = Settings()  # type: ignore

import logging
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(validate_default=True)

    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/1"
    openai_api_key: Optional[str] = None
    replicate_api_token: Optional[str] = None
    hf_home: str = ""
    comfy_api_url: Optional[str] = None
    ddiffusion_storage_directory: str = "/STORAGE"

    @property
    def storage_dir(self) -> str:
        subdir = self.ddiffusion_storage_directory
        os.makedirs(subdir, exist_ok=True)
        return subdir


settings = Settings()  # type: ignore

import os
from typing import Literal, TypeGuard

from common.logger import logger

_VALID_PRECISIONS = (4, 8, 16)


def _read_precision(env_name: str, default: Literal[4, 8, 16] = 4) -> Literal[4, 8, 16]:
    try:
        val = int(os.getenv(env_name, str(default)))
    except ValueError:
        logger.warning("Invalid %s=%r, falling back to %d", env_name, os.getenv(env_name), default)
        return default
    if val not in _VALID_PRECISIONS:
        logger.warning("Unsupported %s=%d, falling back to %d", env_name, val, default)
        return default

    def _is_valid_precision(val: int) -> TypeGuard[Literal[4, 8, 16]]:
        return val in _VALID_PRECISIONS

    if _is_valid_precision(val):
        return val
    return default


# Container/process-level defaults for transformer quantization precision
IMAGE_TRANSFORMER_PRECISION = _read_precision("IMAGE_TRANSFORMER_PRECISION", default=8)
VIDEO_TRANSFORMER_PRECISION = _read_precision("VIDEO_TRANSFORMER_PRECISION", default=4)


# Add boolean env flags to control CPU offloading behavior at container/process level
def _read_bool(env_name: str, default: bool = False) -> bool:
    val = os.getenv(env_name)
    if val is None:
        return default
    return str(val).lower() in ("1", "true", "yes", "y")


# When true, pipelines/models should prefer CPU offload where supported to reduce GPU memory
IMAGE_CPU_OFFLOAD = _read_bool("IMAGE_CPU_OFFLOAD", default=False)
VIDEO_CPU_OFFLOAD = _read_bool("VIDEO_CPU_OFFLOAD", default=False)

ONE_MB_IN_BYTES = 1 * 1024 * 1024

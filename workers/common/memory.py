import gc
import subprocess

import torch

from common.comfy import comfy_utils
from common.logger import logger


def _get_total_gpu_usage():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"], encoding="utf-8"
        )
        return float(result.strip()) / 1024  # Convert MB to GB
    except Exception as e:
        logger.warning(f"Failed to get total system GPU usage: {e}")
        return None


def _get_gpu_memory_usage():
    reserved = torch.cuda.memory_reserved() / 1e9
    allocated = torch.cuda.memory_allocated() / 1e9
    available, total = torch.cuda.mem_get_info()
    system_used = _get_total_gpu_usage()
    if system_used is None:
        used = (total - available) / 1e9
    else:
        used = system_used

    total = total / 1e9
    usage_percent = (used / total) * 100

    return (
        total,
        used,
        reserved,
        allocated,
        usage_percent,
    )


def _get_gpu_memory_usage_pretty():
    total, used, reserved, allocated, usage_percent = _get_gpu_memory_usage()

    return (
        f"GPU Memory Usage: {used:.2f}GB / {total:.2f}GB,  "
        f"Reserved: {reserved:.2f}GB, "
        f"Allocated: {allocated:.2f}GB, "
        f"Usage: {usage_percent:.2f}%"
    )


def _should_free_gpu_memory(threshold_percent: float = 1):
    total, used, reserved, allocated, usage_percent = _get_gpu_memory_usage()
    logger.warning(f"{_get_gpu_memory_usage_pretty()}")
    return usage_percent > threshold_percent


def free_gpu_memory(threshold_percent: float = 25):
    if (_should_free_gpu_memory(threshold_percent=threshold_percent) == False) or (torch.cuda.is_available() == False):
        return

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    comfy_utils.free_resources(unload_models=True, free_memory=False)

    logger.warning(f"Clean {_get_gpu_memory_usage_pretty()}")

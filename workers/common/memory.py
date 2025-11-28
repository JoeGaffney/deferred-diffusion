import gc

import torch

from common.logger import logger

GB_BINARY = 1024**3
GB_DECIMAL = 1e9


def confirm_cuda_available() -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return True


def _get_gpu_memory_usage():
    reserved = torch.cuda.memory_reserved() / GB_BINARY
    allocated = torch.cuda.memory_allocated() / GB_BINARY
    available, total = torch.cuda.mem_get_info()
    used = (total - available) / GB_BINARY

    total = total / GB_BINARY
    usage_percent = (used / total) * 100
    allocated_percent = (allocated / total) * 100

    return (total, used, reserved, allocated, usage_percent, allocated_percent)


def _get_gpu_memory_usage_pretty():
    total, used, reserved, allocated, usage_percent, allocated_percent = _get_gpu_memory_usage()

    return (
        f"GPU Memory Usage: {used:.2f}GiB / {total:.2f}GiB,  "
        f"Reserved: {reserved:.2f}GiB, "
        f"Allocated: {allocated:.2f}GiB, "
        f"Usage: {usage_percent:.2f}%, "
        f"Allocated Percent: {allocated_percent:.2f}%"
    )


def gpu_memory_usage():
    logger.info(f"{_get_gpu_memory_usage_pretty()}")


def free_gpu_memory(message: str = "Cleaned GPU memory"):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    logger.warning(f"{message}: {_get_gpu_memory_usage_pretty()}")


def get_gpu_memory(device: str | torch.device = "cuda") -> float:
    """
    Get the total memory of the current GPU in GiB.

    Args:
        device (str | torch.device): The GPU device to query. Defaults to "cuda".
    """
    if isinstance(device, str):
        device = torch.device(device)
    memory = torch.cuda.get_device_properties(device).total_memory
    return round(memory / GB_BINARY, 2)


def is_memory_exceeded(estimated_memory_gib: float) -> bool:
    available = get_gpu_memory()
    result = estimated_memory_gib >= available

    if result:
        logger.warning(f"Estimated size {estimated_memory_gib}GiB exceeds GPU memory {available}GiB")
    else:
        logger.info(f"Estimated size {estimated_memory_gib}GiB fits within GPU memory {available}GiB")

    return result

from collections import OrderedDict
from functools import wraps
from typing import Any

import torch
from diffusers import DiffusionPipeline

from common.logger import logger

# Global cache for prompt embeddings
# Key is (pipeline_type, args, kwargs)
GLOBAL_PROMPT_CACHE: OrderedDict[Any, Any] = OrderedDict()
MAX_PROMPT_CACHE_SIZE = 64


def _move_to_device(obj: Any, device):
    """Recursively move tensors in a nested structure to a device."""
    if isinstance(obj, torch.Tensor):
        # NOTE important we must detach and clone tensors before caching them
        return obj.detach().clone().to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(x, device) for x in obj)

    # Dict support removed as encode_prompt does not return dicts
    return obj


def clear_global_prompt_cache():
    """Clear the global prompt embeddings cache."""
    GLOBAL_PROMPT_CACHE.clear()
    logger.debug("Global prompt cache cleared")


def get_prompt_cache_if_exists(cache_key):
    """
    Retrieve cached result if it exists and move to Most Recently Used.
    Note: The caller is responsible for moving the result to the correct device.
    """
    if cache_key in GLOBAL_PROMPT_CACHE:
        GLOBAL_PROMPT_CACHE.move_to_end(cache_key)
        logger.info(f"Using cached prompt embeddings for {cache_key[0]}")
        return GLOBAL_PROMPT_CACHE[cache_key]
    return None


def add_prompt_cache(cache_key, result):
    """
    Add a result to the global cache and manage its size.
    Automatically moves the result to CPU to save VRAM.
    """
    # Move to CPU for storage
    cpu_result = _move_to_device(result, "cpu")

    GLOBAL_PROMPT_CACHE[cache_key] = cpu_result
    if len(GLOBAL_PROMPT_CACHE) > MAX_PROMPT_CACHE_SIZE:
        GLOBAL_PROMPT_CACHE.popitem(last=False)  # Remove Least Recently Used

    logger.info(
        f"Prompt cached for {cache_key[0]}. Current cache size: ({len(GLOBAL_PROMPT_CACHE)}/{MAX_PROMPT_CACHE_SIZE})"
    )


def make_hashable(obj):
    if isinstance(obj, (list, tuple)):
        return tuple(make_hashable(i) for i in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    return obj


def enable_prompt_caching(pipeline: DiffusionPipeline) -> DiffusionPipeline:
    """
    Generic wrapper to cache the results of encode_prompt on any diffusers pipeline.
    Uses a global cache shared across pipeline instances to avoid redundant text encoding
    even when pipelines are reloaded.
    """
    if not hasattr(pipeline, "encode_prompt"):
        logger.warning("Pipeline does not have encode_prompt method; cannot enable prompt caching")
        return pipeline

    if hasattr(pipeline, "_prompt_cache_enabled"):
        return pipeline  # Already enabled

    original_encode_prompt = pipeline.encode_prompt
    pipeline_identity = pipeline.__class__.__name__

    @wraps(original_encode_prompt)
    def wrapped_encode_prompt(*args, **kwargs):
        try:
            cache_key = (pipeline_identity, make_hashable(args), make_hashable(kwargs))
        except (TypeError, ValueError):
            logger.warning("Failed to create hashable cache key; skipping prompt caching")
            return original_encode_prompt(*args, **kwargs)

        cached_result = get_prompt_cache_if_exists(cache_key)
        if cached_result is not None:
            # Move back to the target device (e.g. CUDA)
            target_device = kwargs.get("device") or getattr(pipeline, "device", torch.device("cuda"))
            return _move_to_device(cached_result, target_device)

        result = original_encode_prompt(*args, **kwargs)

        add_prompt_cache(cache_key, result)

        return result

    # Monkey patch the instance method
    pipeline.encode_prompt = wrapped_encode_prompt
    pipeline._prompt_cache_enabled = True  # type: ignore[attr-defined]
    return pipeline

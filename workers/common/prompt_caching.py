from collections import OrderedDict
from functools import wraps

import torch
from diffusers import DiffusionPipeline

from common.logger import logger

# Global cache for prompt embeddings
# Key is (pipeline_type, args, kwargs)
GLOBAL_PROMPT_CACHE = OrderedDict()
MAX_PROMPT_CACHE_SIZE = 64


def clear_global_prompt_cache():
    """Clear the global prompt embeddings cache."""
    GLOBAL_PROMPT_CACHE.clear()
    logger.debug("Global prompt cache cleared")


def get_prompt_cache_total_mb() -> float:
    """
    Calculate the total memory usage of the prompt cache in Megabytes (MB).
    Each element in GLOBAL_PROMPT_CACHE can be a Tensor or a tuple/list of Tensors.
    """
    total_bytes = 0

    def get_size(obj):
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, (list, tuple)):
            return sum(get_size(i) for i in obj)
        return 0

    for value in GLOBAL_PROMPT_CACHE.values():
        total_bytes += get_size(value)

    return total_bytes / (1024 * 1024)


def get_prompt_cache_if_exists(cache_key):
    """
    Retrieve cached result if it exists and move to Most Recently Used.
    """
    if cache_key in GLOBAL_PROMPT_CACHE:
        GLOBAL_PROMPT_CACHE.move_to_end(cache_key)
        logger.info(f"Using cached prompt embeddings for {cache_key[0]}")
        return GLOBAL_PROMPT_CACHE[cache_key]
    return None


def add_prompt_cache(cache_key, result):
    """
    Add a result to the global cache and manage its size.
    """
    GLOBAL_PROMPT_CACHE[cache_key] = result
    if len(GLOBAL_PROMPT_CACHE) > MAX_PROMPT_CACHE_SIZE:
        GLOBAL_PROMPT_CACHE.popitem(last=False)  # Remove Least Recently Used

    mb = get_prompt_cache_total_mb()
    logger.info(
        f"Prompt cached for {cache_key[0]}. Current cache size: {mb:.2f} MB ({len(GLOBAL_PROMPT_CACHE)}/{MAX_PROMPT_CACHE_SIZE})"
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
        logger.warning("Prompt caching already enabled on this pipeline instance")
        return pipeline  # Already enabled

    original_encode_prompt = pipeline.encode_prompt
    pipeline_identity = pipeline.__class__.__name__

    @wraps(original_encode_prompt)
    def wrapped_encode_prompt(*args, **kwargs):
        print("wrapped_encode_prompt called")
        try:
            # Create a cache key from identity and hashable representation of all arguments
            # Identity ensures we don't use Flux embeddings for a Wan model, etc.
            cache_key = (pipeline_identity, make_hashable(args), make_hashable(kwargs))
        except (TypeError, ValueError):
            logger.warning("Failed to create hashable cache key; skipping prompt caching")
            # Fallback: if something isn't hashable, just compute normally
            return original_encode_prompt(*args, **kwargs)

        cached_result = get_prompt_cache_if_exists(cache_key)
        if cached_result is not None:
            return cached_result

        # Compute new results (e.g., prompt_embeds, negative_prompt_embeds)
        result = original_encode_prompt(*args, **kwargs)

        # Store in global cache
        add_prompt_cache(cache_key, result)

        return result

    # Monkey patch the instance method
    pipeline.encode_prompt = wrapped_encode_prompt
    pipeline._prompt_cache_enabled = True
    return pipeline

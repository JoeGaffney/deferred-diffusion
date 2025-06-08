import os
from typing import Literal

import torch

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix multiplications

import gc
import time
from collections import OrderedDict
from functools import wraps

import psutil
import torch
from cachetools.keys import hashkey
from transformers import BitsAndBytesConfig, TorchAoConfig

from common.logger import logger
from utils.utils import time_info_decorator


class ModelLRUCache:
    def __init__(self, max_models=2):
        self.cache = OrderedDict()
        self.max_models = max_models
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_load(self, key, loader_fn):
        if key in self.cache:
            # Move to end (most recently used position)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit for {key}")
            return self.cache[key]

        self.misses += 1

        # Evict least recently used model if at capacity
        if len(self.cache) >= self.max_models:
            aggressive_eviction = True
            if aggressive_eviction:
                self._evict_all()
            else:
                self._evict_lru()

        start = time.time()
        pipeline = loader_fn()
        self.cache[key] = pipeline
        end = time.time()
        logger.warning(
            f"Cache miss for {key} - took: {end - start:.2f}s - Cache size: {len(self.cache)}/{self.max_models}"
        )

        return pipeline

    def _evict_lru(self):
        if not self.cache:
            return

        # Get the first item (least recently used)
        oldest_key, oldest_pipeline = next(iter(self.cache.items()))

        logger.warning(f"Evicting LRU model: {oldest_key}")
        self._cleanup(oldest_pipeline)
        self.cache.popitem(last=False)  # Remove from the beginning (LRU)
        self.evictions += 1

    def _evict_all(self):
        if not self.cache:
            return

        cache_size = len(self.cache)
        logger.warning(f"Evicting all {cache_size} models from cache")

        # Clean up all pipelines
        for key, pipeline in list(self.cache.items()):
            try:
                self._cleanup(pipeline)
            except Exception as e:
                logger.error(f"Error cleaning up pipeline {key}: {e}")

        # Clear the cache
        self.cache.clear()
        self.evictions += cache_size

        # Force additional garbage collection
        gc.collect()

    def _cleanup(self, pipeline):
        try:
            gc.collect()
            del pipeline
            gc.collect()
        except Exception as e:
            logger.error(f"Pipeline cleanup error: {e}")

    def get_stats(self):
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "current_cache_size": len(self.cache),
        }

    def reset_cache(self):
        """Reset the cache and clear all models."""
        self._evict_all()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        logger.info("Cache reset successfully.")


# Global cache
global_pipeline_cache = ModelLRUCache(max_models=2)  # Adjust max_models as needed


def reset_global_pipeline_cache():
    """Reset the global pipeline cache."""
    global_pipeline_cache.reset_cache()


def decorator_global_pipeline_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashkey(func.__name__, *args, **kwargs)
        return global_pipeline_cache.get_or_load(key, lambda: func(*args, **kwargs))

    return wrapper


def optimize_pipeline(pipe, disable_safety_checker=True, sequential_cpu_offload=False):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    # Enable CPU offload to save GPU memory
    if sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass  # VAE tiling is not available for all models

    # NOTE Breaks adapter workflows
    # pipe.enable_attention_slicing("auto")  # Enable attention slicing for faster inference
    if disable_safety_checker:
        pipe.safety_checker = dummy_safety_checker

    return pipe


def get_quant_dir(model_id: str, subfolder: str, load_in_4bit: bool) -> str:
    quant_bit = "4bit" if load_in_4bit else "8bit"
    subfolder_name = "default" if subfolder == "" else subfolder
    hf_home = os.getenv("HF_HOME", "")
    quant_dir = os.path.join(hf_home, "quantized", model_id, quant_bit, subfolder_name)
    return os.path.normpath(quant_dir)


@time_info_decorator
def get_quantized_model(
    model_id, subfolder, model_class, target_precision: Literal[4, 8, 16] = 8, torch_dtype=torch.float16
):
    """
    Load a quantized model component if available locally; otherwise, load original,
    quantize, save locally, and return.

    Args:
        model_id (str): Hugging Face repo/model ID.
        subfolder (str): Subfolder name for the model component (e.g., "transformer").
        model_class (class): The HF model class to load (e.g., WanTransformer3DModel).
        target_precision (Literal[4, 8, 16]): Target precision for quantization.
        torch_dtype (torch.dtype): Dtype to use when loading.

    Returns:
        model instance
    """

    if target_precision == 16:
        logger.warning(f"Quantization disabled for {model_id} subfolder {subfolder}")
        return model_class.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch_dtype)

    load_in_4bit = target_precision == 4
    quant_dir = get_quant_dir(model_id, subfolder, load_in_4bit)

    # NOTE does not support CPU model offload so using TorchAo for 8bit
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)  # , llm_int8_enable_fp32_cpu_offload=True)
    quant_config = TorchAoConfig("int8_weight_only")
    use_safetensors = False
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype
        )

    try:
        logger.info(f"Loading quantized model from {quant_dir}")
        model = model_class.from_pretrained(
            quant_dir, torch_dtype=torch_dtype, local_files_only=True, use_safetensors=use_safetensors
        )
    except Exception as e:
        logger.error(f"Failed to load quantized model from {quant_dir}: {e}")
        logger.info(f"Loading and quantizing {model_id} subfolder {subfolder}")
        model = model_class.from_pretrained(
            model_id, subfolder=subfolder, quantization_config=quant_config, torch_dtype=torch_dtype
        )
        os.makedirs(quant_dir, exist_ok=True)
        model.save_pretrained(quant_dir, safe_serialization=use_safetensors)
        logger.info(f"Saved quantized model to {quant_dir}")

    return model

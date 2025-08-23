import gc
import os
import time
from collections import OrderedDict
from functools import lru_cache, wraps
from typing import Literal

import torch
from accelerate.hooks import CpuOffload
from cachetools.keys import hashkey
from diffusers import GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    T5EncoderModel,
    TorchAoConfig,
    UMT5EncoderModel,
)

from common.logger import logger
from common.memory import free_gpu_memory, gpu_memory_usage
from utils.utils import time_info_decorator

torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix multiplications


# Keep reference to original (if you want to restore later)
_original_pre_forward = CpuOffload.pre_forward

# depends on systems ram resources how many models can be safely cached in ram
MAX_MODEL_CACHE = int(os.getenv("MAX_MODEL_CACHE", 2))


@time_info_decorator
def patched_pre_forward(self, module, *args, **kwargs):
    return _original_pre_forward(self, module, *args, **kwargs)


# Apply patch
CpuOffload.pre_forward = patched_pre_forward


class ModelLRUCache:
    def __init__(self, max_models=1):
        self.cache = OrderedDict()
        self.max_models = max_models
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_load(self, key, loader_fn):
        # Ensure we have enough free GPU memory before loading a new model
        free_gpu_memory()

        if key in self.cache:
            # Move to end (most recently used position)
            self.cache.move_to_end(key)
            self.hits += 1
            logger.debug(f"Cache hit for {key}")
            return self.cache[key]

        self.misses += 1

        # Evict least recently used model if at capacity
        if len(self.cache) >= self.max_models:
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

    def _cleanup(self, pipeline):
        try:
            pipeline.to("cpu")
        except Exception as e:
            logger.error(f"Error moving pipeline to CPU: {e}")

        try:
            gc.collect()
            del pipeline
            free_gpu_memory(threshold_percent=1)
        except Exception as e:
            logger.error(f"Pipeline cleanup error: {e}")

    def get_stats(self):
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "current_cache_size": len(self.cache),
        }


# Global cache
global_pipeline_cache = ModelLRUCache(max_models=1)


def decorator_global_pipeline_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashkey(func.__name__, *args, **kwargs)
        return global_pipeline_cache.get_or_load(key, lambda: func(*args, **kwargs))

    return wrapper


@time_info_decorator
def optimize_pipeline(pipe, disable_safety_checker=True, offload=True, vae_tiling=True):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if vae_tiling:
        try:
            pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
            pipe.vae.enable_slicing()
        except:
            pass  # VAE tiling is not available for all models

    if disable_safety_checker:
        pipe.safety_checker = dummy_safety_checker

    gpu_memory_usage()
    return pipe


# NOTE: Currently unused – kept for reference in case GGUF support is needed in future
@time_info_decorator
def get_gguf_model(
    repo_id: str, filename: str, model_class, subfolder: str = "", config=None, torch_dtype=torch.bfloat16
):
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    args = {}
    if subfolder != "":
        args["subfolder"] = subfolder
    if config:
        args["config"] = config

    return model_class.from_single_file(
        path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
        torch_dtype=torch_dtype,
        **args,
    )


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
    quant_dir = get_quant_dir(model_id, subfolder, load_in_4bit=load_in_4bit)

    # NOTE possibly optium quanto is better for 8bit
    # bits and bytes does not offload with 8bit
    quant_config = TorchAoConfig("int8_weight_only")
    use_safetensors = False
    if load_in_4bit:
        # Use BitsAndBytesConfig for 4-bit quantization
        use_safetensors = True
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
            model_id,
            subfolder=subfolder,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
        )
        os.makedirs(quant_dir, exist_ok=True)
        model.save_pretrained(quant_dir, safe_serialization=use_safetensors)
        logger.info(f"Saved quantized model to {quant_dir}")

    return model


# Cache this one as used in many pipelines
@lru_cache(maxsize=1)
def get_quantized_t5_text_encoder(target_precision) -> T5EncoderModel:
    T5_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"

    return get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",
        model_class=T5EncoderModel,
        target_precision=target_precision,
        torch_dtype=torch.bfloat16,
    )


def get_quantized_umt5_text_encoder(target_precision) -> UMT5EncoderModel:
    UMT_T5_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

    return get_quantized_model(
        model_id=UMT_T5_MODEL_PATH,
        subfolder="text_encoder",
        model_class=UMT5EncoderModel,
        target_precision=target_precision,
        torch_dtype=torch.bfloat16,
    )


def get_quantized_qwen_2_5_text_encoder(target_precision) -> Qwen2_5_VLForConditionalGeneration:
    return get_quantized_model(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        subfolder="",
        model_class=Qwen2_5_VLForConditionalGeneration,
        target_precision=target_precision,
        torch_dtype=torch.bfloat16,
    )

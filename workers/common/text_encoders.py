import gc
import time
from collections import OrderedDict
from functools import wraps

import torch
from cachetools import LRUCache
from cachetools.keys import hashkey
from diffusers import (
    FluxPipeline,
    LTXConditionPipeline,
    QwenImageEditPlusPipeline,
    QwenImagePipeline,
    StableDiffusion3Pipeline,
    WanPipeline,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    T5EncoderModel,
    UMT5EncoderModel,
)

from common.logger import logger
from common.pipeline_helpers import time_info_decorator

GLOBAL_PROMPT_CACHE_SIZE = 512
global_prompt_cache = LRUCache(maxsize=GLOBAL_PROMPT_CACHE_SIZE)


def get_prompt_from_cache(model_name, prompt):
    key = hashkey(model_name, prompt)
    return global_prompt_cache.get(key)


def set_prompt_in_cache(model_name, prompt, embedding_tuple):
    key = hashkey(model_name, prompt)
    global_prompt_cache[key] = embedding_tuple


# NOTE import we must detach and clone tensors before caching them
def convert_tensor(tensor, device="cuda", dtype=torch.bfloat16):
    if tensor is None:
        return None
    t = tensor.detach().clone().to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


class ModelLRUCache:
    def __init__(self, max_models=1):
        self.cache = OrderedDict()
        self.max_models = max_models

    def get_or_load(self, key, loader_fn):
        if key in self.cache:
            # Move to end (most recently used position)
            self.cache.move_to_end(key)
            logger.debug(f"Cache hit for {key}")
            return self.cache[key]

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

        logger.debug(f"Evicting TextEncoder LRU model: {oldest_key}")
        self._cleanup(oldest_pipeline)
        self.cache.popitem(last=False)  # Remove from the beginning (LRU)

    def _cleanup(self, pipeline):
        gc.collect()
        try:
            del pipeline
            gc.collect()
        except Exception as e:
            logger.error(f"Pipeline cleanup error: {e}")


# Global cache
global_text_encoder_cache = ModelLRUCache(max_models=1)


def decorator_global_text_encoder_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashkey(func.__name__, *args, **kwargs)
        return global_text_encoder_cache.get_or_load(key, lambda: func(*args, **kwargs))

    return wrapper


@decorator_global_text_encoder_cache
def _pipeline_wan_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        text_encoder=UMT5EncoderModel.from_pretrained(
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            subfolder="text_encoder",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        transformer_2=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)


@time_info_decorator
def wan_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None

    cached = get_prompt_from_cache("wan", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_wan_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=256,
    )
    if prompt_embeds is not None:
        prompt_embeds = convert_tensor(prompt_embeds)

    set_prompt_in_cache("wan", prompt, prompt_embeds)
    return prompt_embeds


@decorator_global_text_encoder_cache
def _pipeline_flux_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        text_encoder_2=T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
        image_encoder=None,
        feature_extractor=None,
    ).to(device)


@time_info_decorator
def flux_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None, None

    cached = get_prompt_from_cache("flux", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_flux_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        max_sequence_length=256,
    )
    if prompt_embeds is not None:
        prompt_embeds = convert_tensor(prompt_embeds)
        pooled_prompt_embeds = convert_tensor(pooled_prompt_embeds)

    set_prompt_in_cache("flux", prompt, (prompt_embeds, pooled_prompt_embeds))
    return prompt_embeds, pooled_prompt_embeds


@decorator_global_text_encoder_cache
def _pipeline_sd3_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        text_encoder_3=T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        vae=None,
        scheduler=None,
        image_encoder=None,
        feature_extractor=None,
        torch_dtype=torch_dtype,
    ).to(device)


@time_info_decorator
def sd3_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None, None

    cached = get_prompt_from_cache("sd3", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_sd3_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        do_classifier_free_guidance=False,
        max_sequence_length=256,
    )
    prompt_embeds = convert_tensor(prompt_embeds)
    pooled_prompt_embeds = convert_tensor(pooled_prompt_embeds)

    set_prompt_in_cache("sd3", prompt, (prompt_embeds, pooled_prompt_embeds))
    return prompt_embeds, pooled_prompt_embeds


@decorator_global_text_encoder_cache
def _pipeline_ltx_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-distilled",
        text_encoder=T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            subfolder="text_encoder_2",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)


@time_info_decorator
def ltx_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None, None

    cached = get_prompt_from_cache("ltx", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_ltx_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, prompt_attention_mask, _, _ = pipe.encode_prompt(
        prompt=prompt,
        max_sequence_length=256,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = convert_tensor(prompt_embeds)
    prompt_attention_mask = convert_tensor(prompt_attention_mask)

    set_prompt_in_cache("ltx", prompt, (prompt_embeds, prompt_attention_mask))
    return prompt_embeds, prompt_attention_mask


@decorator_global_text_encoder_cache
def _pipeline_qwen_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
        text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            subfolder="",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)


@time_info_decorator
def qwen_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None, None

    cached = get_prompt_from_cache("qwen", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_qwen_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=prompt,
        max_sequence_length=256,
    )

    prompt_embeds = convert_tensor(prompt_embeds)
    prompt_embeds_mask = convert_tensor(prompt_embeds_mask, dtype=torch.long)
    set_prompt_in_cache("qwen", prompt, (prompt_embeds, prompt_embeds_mask))
    return prompt_embeds, prompt_embeds_mask


@decorator_global_text_encoder_cache
def _pipeline_qwen_edit_text_encoder(torch_dtype=torch.float32, device="cpu"):
    return QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2509",
        text_encoder=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            subfolder="",
            torch_dtype=torch_dtype,
        ),
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)


@time_info_decorator
def qwen_edit_encode(prompt, torch_dtype=torch.float32, device="cpu"):
    if prompt == "":
        return None, None

    cached = get_prompt_from_cache("qwen", prompt)
    if cached is not None:
        return cached

    pipe = _pipeline_qwen_edit_text_encoder(torch_dtype=torch_dtype, device=device)
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt=prompt,
        max_sequence_length=256,
    )

    prompt_embeds = convert_tensor(prompt_embeds)
    prompt_embeds_mask = convert_tensor(prompt_embeds_mask, dtype=torch.long)
    set_prompt_in_cache("qwen", prompt, (prompt_embeds, prompt_embeds_mask))
    return prompt_embeds, prompt_embeds_mask

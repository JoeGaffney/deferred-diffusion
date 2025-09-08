import gc
import time
from collections import OrderedDict
from functools import lru_cache, wraps

import torch
from cachetools import LRUCache
from cachetools.keys import hashkey
from diffusers import (
    FluxPipeline,
    LTXConditionPipeline,
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

prompt_cache = LRUCache(maxsize=512)


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

        logger.warning(f"Evicting LRU model: {oldest_key}")
        self._cleanup(oldest_pipeline)
        self.cache.popitem(last=False)  # Remove from the beginning (LRU)

    def _cleanup(self, pipeline):
        gc.collect()
        try:
            del pipeline.pipe
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
def get_pipeline_wan_text_encoder(torch_dtype=torch.float32, device="cpu"):
    text_encoder = UMT5EncoderModel.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        text_encoder=text_encoder,
        transformer=None,
        transformer_2=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: WanPipeline):
            self.pipe = pipe
            self.cache = {}

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=256):
            if prompt == "":
                return None

            key = hashkey(prompt)
            if key in self.cache:
                return self.cache[key]

            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=256,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)

            self.cache[key] = prompt_embeds
            return prompt_embeds

    return TextEncoderWrapper(pipe)


@decorator_global_text_encoder_cache
def get_pipeline_flux_text_encoder(torch_dtype=torch.float32, device="cpu"):
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        text_encoder_2=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: FluxPipeline):
            self.pipe = pipe
            self.cache = {}

        @time_info_decorator
        def encode(self, prompt):
            if prompt == "":
                return None, None
            key = hashkey(prompt)
            if key in self.cache:
                return self.cache[key]

            prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=512,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device="cuda", dtype=torch.bfloat16)

            self.cache[key] = (prompt_embeds, pooled_prompt_embeds)
            return prompt_embeds, pooled_prompt_embeds

    return TextEncoderWrapper(pipe)


@decorator_global_text_encoder_cache
def get_pipeline_sd3_text_encoder(torch_dtype=torch.float32, device="cpu"):
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        text_encoder_3=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        image_encoder=None,
        feature_extractor=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: StableDiffusion3Pipeline):
            self.pipe = pipe
            self.cache = {}

        @time_info_decorator
        def encode(self, prompt):
            if prompt == "":
                return None, None

            key = hashkey(prompt)
            if key in self.cache:
                return self.cache[key]

            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                do_classifier_free_guidance=False,
                max_sequence_length=256,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device="cuda", dtype=torch.bfloat16)

            self.cache[key] = (prompt_embeds, pooled_prompt_embeds)
            return prompt_embeds, pooled_prompt_embeds

    return TextEncoderWrapper(pipe)


@decorator_global_text_encoder_cache
def get_pipeline_ltx_text_encoder(torch_dtype=torch.float32, device="cpu"):
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    pipe = LTXConditionPipeline.from_pretrained(
        "Lightricks/LTX-Video-0.9.7-distilled",
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: LTXConditionPipeline):
            self.pipe = pipe
            self.cache = {}

        @time_info_decorator
        def encode(self, prompt):
            if prompt == "":
                return None, None

            key = hashkey(prompt)
            if key in self.cache:
                return self.cache[key]

            prompt_embeds, prompt_attention_mask, _, _ = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=256,
                do_classifier_free_guidance=False,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                prompt_attention_mask = prompt_attention_mask.to(device="cuda", dtype=torch.bfloat16)

            self.cache[key] = (prompt_embeds, prompt_attention_mask)
            return prompt_embeds, prompt_attention_mask

    return TextEncoderWrapper(pipe)


@decorator_global_text_encoder_cache
def get_pipeline_qwen_text_encoder(torch_dtype=torch.float32, device="cpu"):
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        subfolder="",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    pipe = QwenImagePipeline.from_pretrained(
        "ovedrive/qwen-image-4bit",
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: QwenImagePipeline):
            self.pipe = pipe
            self.cache = {}

        @time_info_decorator
        def encode(self, prompt):
            if prompt == "":
                return None, None

            key = hashkey(prompt)
            if key in self.cache:
                return self.cache[key]

            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=1024,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                prompt_embeds_mask = prompt_embeds_mask.to(device=device)
                if prompt_embeds_mask.dtype != torch.long:
                    prompt_embeds_mask = prompt_embeds_mask.long()

            self.cache[key] = (prompt_embeds, prompt_embeds_mask)
            return prompt_embeds, prompt_embeds_mask

    return TextEncoderWrapper(pipe)

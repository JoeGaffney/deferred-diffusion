from functools import lru_cache

import torch
from diffusers import FluxPipeline, LTXConditionPipeline, QwenImagePipeline, WanPipeline
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    T5EncoderModel,
    UMT5EncoderModel,
)

from common.pipeline_helpers import time_info_decorator


@lru_cache(maxsize=1)
@time_info_decorator
def get_pipeline_wan_text_encoder(
    model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", torch_dtype=torch.float32, device="cpu"
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = WanPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=None,
        transformer_2=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: WanPipeline):
            self.pipe = pipe

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=256):

            device = self.pipe.device
            dtype = self.pipe.dtype

            prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=256,
                device=device,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
            return prompt_embeds

    return TextEncoderWrapper(pipe)


@lru_cache(maxsize=1)
@time_info_decorator
def get_pipeline_flux_text_encoder(
    model_id="black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float32, device="cpu"
):
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: FluxPipeline):
            self.pipe = pipe

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=512):

            device = self.pipe.device
            dtype = self.pipe.dtype

            prompt_embeds, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                pooled_prompt_embeds = pooled_prompt_embeds.to(device="cuda", dtype=torch.bfloat16)

            return prompt_embeds, pooled_prompt_embeds

    return TextEncoderWrapper(pipe)


@lru_cache(maxsize=1)
@time_info_decorator
def get_pipeline_ltx_text_encoder(
    model_id="Lightricks/LTX-Video-0.9.7-distilled", torch_dtype=torch.float32, device="cpu"
):
    text_encoder = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: LTXConditionPipeline):
            self.pipe = pipe

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=256):
            if prompt == "":
                return None, None

            device = self.pipe.device
            dtype = self.pipe.dtype

            prompt_embeds, prompt_attention_mask, _, _ = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
                do_classifier_free_guidance=False,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                prompt_attention_mask = prompt_attention_mask.to(device="cuda", dtype=torch.bfloat16)

            return prompt_embeds, prompt_attention_mask

    return TextEncoderWrapper(pipe)


@lru_cache(maxsize=1)
@time_info_decorator
def get_pipeline_qwen_text_encoder(model_id="ovedrive/qwen-image-4bit", torch_dtype=torch.float32, device="cpu"):
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        subfolder="",
        torch_dtype=torch_dtype,
    ).to(device)

    pipe = QwenImagePipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=None,
        vae=None,
        scheduler=None,
        torch_dtype=torch_dtype,
    ).to(device)

    class TextEncoderWrapper:
        def __init__(self, pipe: QwenImagePipeline):
            self.pipe = pipe

        @time_info_decorator
        @lru_cache(maxsize=5)
        def encode(self, prompt, max_sequence_length=1024):

            device = self.pipe.device
            dtype = self.pipe.dtype

            prompt_embeds, prompt_embeds_mask = self.pipe.encode_prompt(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(device="cuda", dtype=torch.bfloat16)
                prompt_embeds_mask = prompt_embeds_mask.to(device=device)
                if prompt_embeds_mask.dtype != torch.long:
                    prompt_embeds_mask = prompt_embeds_mask.long()

            return prompt_embeds, prompt_embeds_mask

    return TextEncoderWrapper(pipe)

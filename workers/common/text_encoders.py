from functools import lru_cache

import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, UMT5EncoderModel

from common.memory import LOW_VRAM
from common.pipeline_helpers import time_info_decorator


class TextEncoderManager:
    """We keep seprate encode methods for control different variants in pipelines.
    This is the safest way to do it and still be efficient with caching."""

    def __init__(self, model_id, tokenizer_subfolder="tokenizer", textencoder_subfolder="text_encoder", device="cpu"):
        self.torch_dtype = torch.float16
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder=tokenizer_subfolder)
        self.text_encoder = AutoModel.from_pretrained(
            model_id, subfolder=textencoder_subfolder, torch_dtype=self.torch_dtype
        ).to(device)

    @lru_cache(maxsize=5)
    @time_info_decorator
    @torch.no_grad()
    def encode_wan(self, prompt, max_sequence_length=256):
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        text_inputs = self.tokenizer(  # type: ignore
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_embeds = outputs.last_hidden_state.to(dtype=dtype, device=device)
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(dtype)

        return prompt_embeds

    @lru_cache(maxsize=5)
    @time_info_decorator
    @torch.no_grad()
    def encode_flux(self, prompt, max_sequence_length=256):
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds


@time_info_decorator
@lru_cache(maxsize=3)
def _get_text_encoder(model_id, tokenizer_subfolder="tokenizer", textencoder_subfolder="text_encoder", device="cpu"):
    return TextEncoderManager(
        model_id, tokenizer_subfolder=tokenizer_subfolder, textencoder_subfolder=textencoder_subfolder, device=device
    )


def get_umt5_text_encoder() -> TextEncoderManager:
    return _get_text_encoder(
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        tokenizer_subfolder="tokenizer",
        textencoder_subfolder="text_encoder",
        device="cpu",
    )


def get_t5_text_encoder() -> TextEncoderManager:
    return _get_text_encoder(
        "black-forest-labs/FLUX.1-schnel",
        tokenizer_subfolder="tokenizer_2",
        textencoder_subfolder="text_encoder_2",
        device="cpu",
    )


def get_qwen_2_5_text_encoder() -> TextEncoderManager:
    return _get_text_encoder(
        "Qwen/Qwen2.5-VL-7B-Instruct", tokenizer_subfolder="", textencoder_subfolder="", device="cpu"
    )

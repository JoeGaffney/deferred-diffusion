from typing import Literal

import torch
from transformers import (
    Mistral3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    T5EncoderModel,
    UMT5EncoderModel,
)

from common.pipeline_helpers import get_quantized_model


def get_t5_text_encoder() -> T5EncoderModel:
    return get_quantized_model(
        model_id="black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )


def get_qwen2_5_text_encoder(
    target_precision: Literal[4, 8, 16] = 8,
) -> Qwen2_5_VLForConditionalGeneration:
    return get_quantized_model(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        subfolder="",
        model_class=Qwen2_5_VLForConditionalGeneration,
        target_precision=target_precision,
        torch_dtype=torch.bfloat16,
    )


def get_umt5_text_encoder() -> UMT5EncoderModel:
    return get_quantized_model(
        model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        subfolder="text_encoder",
        model_class=UMT5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )


def get_mistral3_text_encoder() -> Mistral3ForConditionalGeneration:
    return get_quantized_model(
        # NOTE just used the one allready in 4bit for now
        # model_id="black-forest-labs/FLUX.2-dev",
        model_id="diffusers/FLUX.2-dev-bnb-4bit",
        subfolder="text_encoder",
        model_class=Mistral3ForConditionalGeneration,
        target_precision=16,
        torch_dtype=torch.bfloat16,
    )

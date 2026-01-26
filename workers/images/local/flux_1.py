from pathlib import Path
from typing import List

import torch
from diffusers import (
    AutoPipelineForText2Image,
    FluxFillPipeline,
    FluxKontextPipeline,
    FluxPipeline,
)
from nunchaku import NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision
from PIL import Image

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    optimize_pipeline,
    task_log_callback,
)
from common.text_encoders import get_t5_text_encoder
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{get_precision()}_r32-flux.1-krea-dev.safetensors"
    )

    pipe = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


@decorator_global_pipeline_cache
def get_kontext_pipeline(model_id):
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
    )

    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id):
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors"
    )

    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


def text_to_image_call(context: ImageContext) -> List[Path]:
    pipe = get_pipeline("black-forest-labs/FLUX.1-Krea-dev")

    processed_image = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        width=context.width,
        height=context.height,
        num_inference_steps=30,
        generator=context.generator,
        guidance_scale=2.5,
        callback_on_step_end=task_log_callback(30),  # type: ignore
    ).images[0]
    return [context.save_output(processed_image, index=0)]


def image_to_image_call(context: ImageContext) -> List[Path]:
    pipe = get_kontext_pipeline("black-forest-labs/FLUX.1-Kontext-dev")

    processed_image = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        width=context.width,
        height=context.height,
        image=context.color_image,
        num_inference_steps=30,
        generator=context.generator,
        guidance_scale=2.0,
        callback_on_step_end=task_log_callback(30),  # type: ignore
    ).images[0]
    return [context.save_output(processed_image, index=0)]


def inpainting_call(context: ImageContext) -> List[Path]:
    pipe = get_inpainting_pipeline("black-forest-labs/FLUX.1-Fill-dev")

    processed_image = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        width=context.width,
        height=context.height,
        image=context.color_image,  # type: ignore
        mask_image=context.mask_image,  # type: ignore
        num_inference_steps=30,
        generator=context.generator,
        guidance_scale=30,
        strength=context.data.strength,
        callback_on_step_end=task_log_callback(30),  # type: ignore
    ).images[0]
    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)
    return text_to_image_call(context)

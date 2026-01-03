from pathlib import Path
from typing import List

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    optimize_pipeline,
    task_log_callback,
)
from images.context import ImageContext

_negative_prompt_default = "worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes"


@decorator_global_pipeline_cache
def get_pipeline(model_id) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    return optimize_pipeline(pipe, offload=is_memory_exceeded(11))


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id) -> StableDiffusionXLInpaintPipeline:
    args = {}
    args["variant"] = "fp16"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        **args,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(11))


def text_to_image_call(context: ImageContext) -> List[Path]:
    pipe = AutoPipelineForText2Image.from_pipe(
        get_pipeline("SG161222/RealVisXL_V4.0"),
        requires_safety_checker=False,
        torch_dtype=torch.bfloat16,
    )

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        num_inference_steps=35,
        generator=context.generator,
        guidance_scale=3.5,
        callback_on_step_end=task_log_callback(35),
    ).images[0]

    return [context.save_output(processed_image, index=0)]


def image_to_image_call(context: ImageContext) -> List[Path]:
    pipe = AutoPipelineForImage2Image.from_pipe(
        get_pipeline("SG161222/RealVisXL_V4.0"),
        requires_safety_checker=False,
        torch_dtype=torch.bfloat16,
    )

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        image=context.color_image,
        num_inference_steps=35,
        generator=context.generator,
        strength=context.data.strength,
        guidance_scale=3.5,
        callback_on_step_end=task_log_callback(35),
    ).images[0]

    return [context.save_output(processed_image, index=0)]


def inpainting_call(context: ImageContext) -> List[Path]:
    pipe = get_inpainting_pipeline("OzzyGT/RealVisXL_V4.0_inpainting")

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        image=context.color_image,  # type: ignore
        mask_image=context.mask_image,  # type: ignore
        num_inference_steps=35,
        generator=context.generator,
        strength=0.95,
        guidance_scale=4.0,
        callback_on_step_end=task_log_callback(35),  # type: ignore
    ).images[0]

    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    context.ensure_divisible(16)

    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)

    return text_to_image_call(context)

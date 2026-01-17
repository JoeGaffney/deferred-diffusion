from pathlib import Path
from typing import List

import torch
from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from common.text_encoders import get_qwen3_8b_text_encoder
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):

    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=Flux2Transformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, text_encoder=get_qwen3_8b_text_encoder(), transformer=transformer, torch_dtype=torch.bfloat16
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(16))


def text_to_image_call(context: ImageContext):
    model_id = "black-forest-labs/FLUX.2-klein-9B"
    pipe = get_pipeline(model_id)
    prompt = context.data.cleaned_prompt

    # gather all possible reference images
    reference_images = []
    if context.mask_image and context.color_image:
        prompt = (
            "Use image 1 for the mask region for inpainting. And use image 2 for the base image only alter the mask region and aim for a seamless blend. "
            + prompt
        )
        reference_images.append(context.mask_image)
        reference_images.append(context.color_image)
    else:
        if context.color_image:
            reference_images.append(context.color_image)

        for current in context.get_reference_images():
            if current is not None:
                reference_images.append(current)

    processed_image = pipe(
        prompt=prompt,
        image=None if len(reference_images) == 0 else reference_images[:3],
        num_inference_steps=4,
        guidance_scale=1.0,
        height=context.height,
        width=context.width,
        generator=context.generator,
        callback_on_step_end=task_log_callback(4),  # type: ignore
    ).images[0]
    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    return text_to_image_call(context)

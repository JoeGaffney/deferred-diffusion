from functools import lru_cache

import torch
from diffusers import StableDiffusionUpscalePipeline

from common.logger import logger
from common.pipeline_helpers import optimize_pipeline
from images.context import ImageContext


@lru_cache(maxsize=1)
def get_pipeline(model_id) -> StableDiffusionUpscalePipeline:
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    logger.warning(f"Loaded pipeline {model_id}")
    return optimize_pipeline(pipe, disable_safety_checker=False)


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No input image provided")

    pipe = get_pipeline(context.data.model_path)
    scale = 4

    processed_image = pipe(
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        image=context.color_image,
        num_inference_steps=context.data.num_inference_steps,
        generator=context.generator,
        guidance_scale=context.data.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image, scale=scale)
    return processed_image

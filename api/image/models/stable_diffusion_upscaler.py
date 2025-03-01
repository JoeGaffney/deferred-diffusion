from functools import lru_cache

import torch
from common.pipeline_helpers import optimize_pipeline
from diffusers import StableDiffusionUpscalePipeline
from image.context import ImageContext
from image.models.diffusers_helpers import upscale_call
from utils.logger import logger


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )

    logger.warning(f"Loaded pipeline {model_id}")
    return optimize_pipeline(pipe, disable_safety_checker=False)


def main(context: ImageContext, mode="upscaler"):
    context.model = "stabilityai/stable-diffusion-x4-upscaler"
    return upscale_call(get_pipeline(context.model), context, scale=4)

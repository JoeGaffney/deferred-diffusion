import os
from functools import lru_cache

import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
    LTXVideoTransformer3DModel,
)
from transformers import CLIPVisionModel, T5EncoderModel, UMT5EncoderModel

from common.logger import logger
from common.memory import LOW_VRAM
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from utils.utils import ensure_divisible, get_16_9_resolution, resize_image
from videos.context import VideoContext

T5_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"


@decorator_global_pipeline_cache
def get_pipeline(model_id):

    if LOW_VRAM:
        logger.warning("Running in low VRAM mode, using 4bit precision for some  models.")

    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=LTXVideoTransformer3DModel,
        target_precision=4 if LOW_VRAM else 8,
        torch_dtype=torch.bfloat16,
    )

    # text_encoder_2 : [T5EncoderModel]
    # T5, specifically the google/t5-v1_1-xxl variant.
    text_encoder = get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",  # This is the subfolder for the T5 encoder in the flux model
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def image_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    width, height = get_16_9_resolution("720p")
    image = context.image
    image = resize_image(image, 32, 1.0, width, height)

    condition1 = LTXVideoCondition(
        image=image,
        frame_index=0,
    )

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        conditions=[condition1],
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        generator=context.get_generator(),
        guidance_scale=context.data.guidance_scale,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def text_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    width, height = get_16_9_resolution("540p")
    width = ensure_divisible(width, divisor=32)
    height = ensure_divisible(height, divisor=32)

    video = pipe.__call__(
        width=width,
        height=height,
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        num_frames=context.data.num_frames,
        num_inference_steps=context.data.num_inference_steps,
        generator=context.get_generator(),
        guidance_scale=context.data.guidance_scale,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def main(context: VideoContext):
    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

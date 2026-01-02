from pathlib import Path
from typing import List

import torch
from diffusers import (
    HunyuanVideo15ImageToVideoPipeline,
    HunyuanVideo15Pipeline,
    HunyuanVideo15Transformer3DModel,
    attention_backend,
)

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from common.text_encoders import get_qwen2_5_text_encoder
from videos.context import VideoContext


@decorator_global_pipeline_cache
def get_pipeline_t2v(model_id) -> HunyuanVideo15Pipeline:
    transformer = get_quantized_model(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
        subfolder="transformer",
        model_class=HunyuanVideo15Transformer3DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = HunyuanVideo15Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=get_qwen2_5_text_encoder(),
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(35))


@decorator_global_pipeline_cache
def get_pipeline_i2v(model_id) -> HunyuanVideo15ImageToVideoPipeline:
    transformer = get_quantized_model(
        model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled",
        subfolder="transformer",
        model_class=HunyuanVideo15Transformer3DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = HunyuanVideo15ImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=get_qwen2_5_text_encoder(),
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(35))


def text_to_video(context: VideoContext) -> list[Path]:
    pipe = get_pipeline_t2v(model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled")

    with attention_backend("flash_hub"):
        output = pipe(
            prompt=context.data.cleaned_prompt,
            width=context.width,
            height=context.height,
            num_inference_steps=35,
            num_frames=context.data.num_frames,
            generator=context.get_generator(),
            # missing in implementation:
            # callback_on_step_end=task_log_callback(20),  # type: ignore
        ).frames[0]

    return [context.save_output(output, fps=24)]


def image_to_video(context: VideoContext) -> List[Path]:
    pipe = get_pipeline_i2v(model_id="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled")

    with attention_backend("flash_varlen_hub"):
        output = pipe(
            prompt=context.data.cleaned_prompt,
            image=context.image,
            num_inference_steps=12,
            num_frames=context.data.num_frames,
            generator=context.get_generator(),
            # missing in implementation:
            # callback_on_step_end=task_log_callback(20),  # type: ignore
        ).frames[0]

    return [context.save_output(output, fps=24)]


def main(context: VideoContext) -> List[Path]:
    context.rescale_to_max_megapixels(1.0)  # limit to 1 megapixel to avoid OOM
    context.ensure_divisible(16)
    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

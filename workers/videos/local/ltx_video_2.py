import os
from pathlib import Path
from typing import List

import torch
from diffusers import LTX2VideoTransformer3DModel
from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline, LTX2Pipeline
from diffusers.pipelines.ltx2.export_utils import encode_video
from transformers import Gemma3ForConditionalGeneration

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from videos.context import VideoContext

# NOTE still WIP not fully integrated/tested as seeing odd results with diffusers

# _negative_prompt = "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
# match comfy ltx-2 negative prompt
_negative_prompt = "blurry, low quality, still frame, frames, watermark, overlay, titles, has blurbox, has subtitles"
_steps = 40


@decorator_global_pipeline_cache
def get_pipeline(model_id) -> LTX2Pipeline:
    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=LTX2VideoTransformer3DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    text_encoder = get_quantized_model(
        model_id,
        subfolder="text_encoder",
        model_class=Gemma3ForConditionalGeneration,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    # Load pipeline with quantized components
    pipe = LTX2Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    # pipe.vae.enable_tiling()
    return optimize_pipeline(pipe, offload=is_memory_exceeded(33), vae_tiling=False)


def save_video_result(context: VideoContext, pipe, video, audio, frame_rate=24) -> List[Path]:
    output_path = context.get_output_path(0)
    print(f"Saving video to {output_path}")

    video = (video * 255).round().astype("uint8")
    video = torch.from_numpy(video)

    encode_video(
        video[0],
        fps=frame_rate,
        audio=audio[0].float().cpu(),
        audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
        output_path=str(output_path),
    )

    return [output_path]


def text_to_video(context: VideoContext) -> List[Path]:
    pipe = get_pipeline("Lightricks/LTX-2")

    frame_rate = 24.0

    video, audio = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt,
        width=context.width,
        height=context.height,
        generator=context.get_generator(),
        num_frames=context.data.num_frames,
        frame_rate=frame_rate,
        num_inference_steps=_steps,
        guidance_scale=4.0,
        output_type="np",
        return_dict=False,
        callback_on_step_end=task_log_callback(_steps),  # type: ignore
    )

    return save_video_result(context, pipe, video, audio, int(frame_rate))


def image_to_video(context: VideoContext) -> List[Path]:
    if not context.image:
        raise ValueError("image is required for image-to-video generation.")

    pipe = LTX2ImageToVideoPipeline.from_pipe(
        get_pipeline("Lightricks/LTX-2"),
        torch_dtype=torch.bfloat16,
    )

    frame_rate = 24.0

    video, audio = pipe.__call__(
        image=context.image,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt,
        width=context.width,
        height=context.height,
        generator=context.get_generator(),
        num_frames=context.data.num_frames,
        frame_rate=frame_rate,
        num_inference_steps=_steps,
        guidance_scale=4.0,
        output_type="np",
        return_dict=False,
        callback_on_step_end=task_log_callback(_steps),  # type: ignore
    )

    return save_video_result(context, pipe, video, audio, int(frame_rate))


def main(context: VideoContext) -> List[Path]:
    context.rescale_to_max_megapixels(1.0)  # limit to 1 megapixel to avoid OOM
    context.ensure_divisible(32)

    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

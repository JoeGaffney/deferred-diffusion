import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
    LTXVideoTransformer3DModel,
)

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from common.text_encoders import get_t5_text_encoder
from videos.context import VideoContext

_negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes"


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    # NOTE don't actually see much difference in look with Q4 vs Q8
    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=LTXVideoTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=get_t5_text_encoder(),
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(23))


def image_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")
    num_frames = context.data.num_frames

    condition1 = LTXVideoCondition(
        image=context.image,
        frame_index=0,
    )
    conditions = [condition1]

    if context.video_frames:
        num_frames = min(num_frames, len(context.video_frames))
        video_condition = LTXVideoCondition(
            image=context.image,
            video=context.video_frames[:num_frames],
            frame_index=0,
        )
        conditions.append(video_condition)

    video = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt,
        width=context.width,
        height=context.height,
        conditions=conditions,
        num_inference_steps=10,
        num_frames=num_frames,
        generator=context.get_generator(),
        guidance_scale=1.0,
        callback_on_step_end=task_log_callback(10),  # type: ignore
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def text_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    video = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt,
        width=context.width,
        height=context.height,
        num_inference_steps=10,
        num_frames=context.data.num_frames,
        generator=context.get_generator(),
        guidance_scale=1.0,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def main(context: VideoContext):
    context.ensure_divisible(32)
    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

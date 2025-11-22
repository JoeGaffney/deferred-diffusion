import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
    LTXVideoTransformer3DModel,
)

from common.config import VIDEO_CPU_OFFLOAD, VIDEO_TRANSFORMER_PRECISION
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from common.text_encoders import ltx_encode
from videos.context import VideoContext

_negative_prompt_default = "worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes"


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    # NOTE don't actually see much difference in look with Q4 vs Q8
    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=LTXVideoTransformer3DModel,
        target_precision=VIDEO_TRANSFORMER_PRECISION,
        torch_dtype=torch.bfloat16,
    )

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch.bfloat16,
    )

    pipe.vae.enable_tiling()
    if VIDEO_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def image_to_video(context: VideoContext):
    prompt_embeds, prompt_attention_mask = ltx_encode(context.data.cleaned_prompt)
    negative_prompt_embeds, negative_prompt_attention_mask = ltx_encode(_negative_prompt_default)
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    condition1 = LTXVideoCondition(
        image=context.image,
        frame_index=0,
    )
    conditions = [condition1]

    if context.video_frames:
        video_condition = LTXVideoCondition(
            image=context.image,
            video=context.video_frames[: context.data.num_frames],
            frame_index=0,
        )
        conditions.append(video_condition)

    video = pipe.__call__(
        width=context.width,
        height=context.height,
        conditions=conditions,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        num_inference_steps=10,
        num_frames=context.data.num_frames,
        generator=context.get_generator(),
        guidance_scale=1.0,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def text_to_video(context: VideoContext):
    prompt_embeds, prompt_attention_mask = ltx_encode(context.data.cleaned_prompt)
    negative_prompt_embeds, negative_prompt_attention_mask = ltx_encode(_negative_prompt_default)
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    video = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
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

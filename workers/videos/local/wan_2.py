import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from common.text_encoders import get_umt5_text_encoder
from videos.context import VideoContext
from videos.local.wan_vace import video_to_video

# Wan gives better results with a default negative prompt
_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


@decorator_global_pipeline_cache
def get_pipeline_t2v(model_id) -> WanPipeline:
    args = {"boundary_ratio": 0.5}  # even split
    transformer = get_quantized_model(
        model_id="magespace/Wan2.2-T2V-A14B-Lightning-Diffusers",
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    transformer_2 = get_quantized_model(
        model_id="magespace/Wan2.2-T2V-A14B-Lightning-Diffusers",
        subfolder="transformer_2",
        model_class=WanTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=get_umt5_text_encoder(),
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch.bfloat16,
        **args,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    return optimize_pipeline(pipe, offload=is_memory_exceeded(35))


@decorator_global_pipeline_cache
def get_pipeline_i2v(model_id) -> WanImageToVideoPipeline:
    # even split gives strange results - try without for now
    args = {"boundary_ratio": 0.5}
    args = {}
    transformer = get_quantized_model(
        model_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    transformer_2 = get_quantized_model(
        model_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers",
        subfolder="transformer_2",
        model_class=WanTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=get_umt5_text_encoder(),
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch.bfloat16,
        **args,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=3.0)

    return optimize_pipeline(pipe, offload=is_memory_exceeded(35))


def text_to_video(context: VideoContext):
    pipe = get_pipeline_t2v(model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    if context.get_mega_pixels() >= 0.9:  # close to 720p or higher
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    output = pipe(
        prompt=context.data.cleaned_prompt,
        width=context.width,
        height=context.height,
        num_inference_steps=8,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
        callback_on_step_end=task_log_callback(8),  # type: ignore
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def image_to_video(context: VideoContext):
    if context.image is None:
        raise ValueError("No input image provided for image-to-video generation")

    pipe = get_pipeline_i2v(model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    if context.get_mega_pixels() >= 0.9:  # close to 720p or higher
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    output = pipe(
        prompt=context.data.cleaned_prompt,
        width=context.width,
        height=context.height,
        image=context.image,
        last_image=context.last_image,  # type: ignore
        num_inference_steps=8,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
        callback_on_step_end=task_log_callback(8),  # type: ignore
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def main(context: VideoContext):
    context.rescale_to_max_megapixels(1.0)  # limit to 1 megapixel to avoid OOM
    context.ensure_divisible(16)
    if context.data.video and context.data.image:
        return video_to_video(context)

    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

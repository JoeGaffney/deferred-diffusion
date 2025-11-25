import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)

from common.config import VIDEO_CPU_OFFLOAD, VIDEO_TRANSFORMER_PRECISION
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from common.text_encoders import wan_encode
from videos.context import VideoContext
from videos.local.wan_vace import video_to_video

# Wan gives better results with a default negative prompt
_negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


@decorator_global_pipeline_cache
def get_pipeline_i2v(model_id, high_noise: bool, torch_dtype=torch.bfloat16) -> WanImageToVideoPipeline:
    # high noise uses both transformers - low noise only seems busted and gives crazy results atm
    transformer = None
    args = {"boundary_ratio": 1.0}
    if high_noise:
        args = {}
        transformer = get_quantized_model(
            model_id=model_id,
            subfolder="transformer",
            model_class=WanTransformer3DModel,
            target_precision=VIDEO_TRANSFORMER_PRECISION,
            torch_dtype=torch_dtype,
        )

    transformer_2 = get_quantized_model(
        model_id=model_id,
        subfolder="transformer_2",
        model_class=WanTransformer3DModel,
        target_precision=VIDEO_TRANSFORMER_PRECISION,
        torch_dtype=torch_dtype,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=None,
        tokenizer=None,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
        **args,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if VIDEO_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


@decorator_global_pipeline_cache
def get_pipeline_t2v(model_id, high_noise: bool, torch_dtype=torch.bfloat16) -> WanPipeline:
    # high noise uses both transformers
    transformer = None
    args = {"boundary_ratio": 1.0}
    if high_noise:
        args = {}
        transformer = get_quantized_model(
            model_id=model_id,
            subfolder="transformer",
            model_class=WanTransformer3DModel,
            target_precision=VIDEO_TRANSFORMER_PRECISION,
            torch_dtype=torch_dtype,
        )

    transformer_2 = get_quantized_model(
        model_id=model_id,
        subfolder="transformer_2",
        model_class=WanTransformer3DModel,
        target_precision=VIDEO_TRANSFORMER_PRECISION,
        torch_dtype=torch_dtype,
    )

    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=None,
        tokenizer=None,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
        **args,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if VIDEO_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def text_to_video(context: VideoContext):
    prompt_embeds = wan_encode(context.data.cleaned_prompt)
    negative_prompt_embeds = wan_encode(_negative_prompt)
    pipe = get_pipeline_t2v(model_id="magespace/Wan2.2-T2V-A14B-Lightning-Diffusers", high_noise=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=context.get_flow_shift())

    output = pipe(
        width=context.width,
        height=context.height,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=8,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def image_to_video(context: VideoContext):
    if context.image is None:
        raise ValueError("No input image provided for image-to-video generation")

    prompt_embeds = wan_encode(context.data.cleaned_prompt)
    negative_prompt_embeds = wan_encode(_negative_prompt)
    pipe = get_pipeline_i2v(model_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers", high_noise=True)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=context.get_flow_shift())

    output = pipe(
        width=context.width,
        height=context.height,
        image=context.image,
        last_image=context.last_image,  # type: ignore
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=8,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def main(context: VideoContext):
    context.ensure_divisible(16)
    if context.data.video and context.data.image:
        return video_to_video(context)

    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

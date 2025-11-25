import PIL.Image
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from common.config import VIDEO_CPU_OFFLOAD, VIDEO_TRANSFORMER_PRECISION
from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_gguf_model,
    get_quantized_model,
)
from common.text_encoders import wan_encode
from videos.context import VideoContext

# Wan VACE gives better results with a default negative prompt
_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


@decorator_global_pipeline_cache
def get_pipeline(model_id, torch_dtype=torch.bfloat16) -> WanVACEPipeline:

    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanVACETransformer3DModel,
        target_precision=VIDEO_TRANSFORMER_PRECISION,
        torch_dtype=torch_dtype,
    )

    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    pipe = WanVACEPipeline.from_pretrained(
        model_id,
        vae=vae,
        transformer=transformer,
        transformer_2=None,
        text_encoder=None,
        tokenizer=None,
        torch_dtype=torch_dtype,
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


def video_to_video(context: VideoContext):
    if context.video_frames is None:
        raise ValueError("No video frames provided for video-to-video generation")

    if context.image is None:
        raise ValueError("No reference image provided for video generation")

    prompt_embeds = wan_encode(context.data.cleaned_prompt)
    negative_prompt_embeds = wan_encode(_negative_prompt)

    pipe = get_pipeline(model_id="Wan-AI/Wan2.1-VACE-14B-diffusers")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=context.get_flow_shift())

    # Adjust num_frames to meet WanVACE requirements and limit to available frames
    num_frames = min(context.data.num_frames, len(context.video_frames))
    num_frames = context.ensure_frames_divisible(num_frames, 4)

    # Use existing video frames, resized to target dimensions
    video_frames = []
    for i in range(num_frames):
        frame_idx = min(i, len(context.video_frames) - 1)
        frame = context.video_frames[frame_idx].resize((context.width, context.height))
        video_frames.append(frame)

    mask_black = PIL.Image.new("L", (context.width, context.height), 0)
    mask_white = PIL.Image.new("L", (context.width, context.height), 255)
    # Create mask for video-to-video: mask all frames for transformation
    mask_frames = [mask_white] * num_frames

    # NOTE atm does not seem to respect the reference image properly
    reference_images = [context.image]

    logger.info(
        f"Prepared {len(video_frames)} video frames and {len(mask_frames)} mask frames for video-to-video. num_frames={num_frames}, reference_images={'provided' if reference_images else 'none'}"
    )

    output = pipe(
        video=video_frames,
        mask=mask_frames,  # type: ignore
        reference_images=reference_images,  # type: ignore
        conditioning_scale=1,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=context.height,
        width=context.width,
        num_frames=num_frames,
        num_inference_steps=12,
        guidance_scale=5.0,
        generator=context.get_generator(),
    )

    # Extract frames from pipeline output
    video_frames_output = getattr(output, "frames", [output])[0]
    processed_path = context.save_video(video_frames_output, fps=16)
    return processed_path

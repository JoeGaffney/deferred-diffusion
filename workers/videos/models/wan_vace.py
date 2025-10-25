import PIL.Image
import torch
from diffusers import AutoencoderKLWan, WanVACEPipeline, WanVACETransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from common.config import VIDEO_CPU_OFFLOAD, VIDEO_TRANSFORMER_PRECISION
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from common.text_encoders import wan_encode
from videos.context import VideoContext


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

    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    # if VIDEO_CPU_OFFLOAD:
    #     pipe.enable_model_cpu_offload()
    # else:
    #     pipe.to("cuda")
    pipe.to("cuda")
    return pipe


# Wan VACE gives better results with a default negative prompt
_negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


def prepare_video_and_mask(
    first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int
):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


def first_last_frame_to_video(context: VideoContext):
    """
    Generate a video from first and last frames with interpolation in between.
    This is the main functionality of WanVACE pipeline.
    """
    if context.image_last_frame is None:
        raise ValueError("No last frame image provided for first-last frame video generation")

    if context.image is None:
        raise ValueError("No first frame image provided for first-last frame video generation")

    prompt_embeds = wan_encode(context.data.prompt)
    negative_prompt_embeds = wan_encode(_negative_prompt)

    pipe = get_pipeline(model_id="Wan-AI/Wan2.1-VACE-14B-diffusers")

    # Prepare video frames and mask for VACE pipeline
    video_frames, mask_frames = prepare_video_and_mask(
        context.image, context.image_last_frame, context.height, context.width, context.data.num_frames
    )

    output = pipe(
        video=video_frames,
        mask=mask_frames,  # type: ignore
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=context.height,
        width=context.width,
        num_frames=context.data.num_frames,
        num_inference_steps=10,
        guidance_scale=5.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def main(context: VideoContext):
    context.ensure_divisible(16)

    # WanVACE requires both first and last frame
    if context.image_last_frame is None:
        raise ValueError(
            "WanVACE pipeline requires a last frame image. Please provide both first and last frame images."
        )

    return first_last_frame_to_video(context)

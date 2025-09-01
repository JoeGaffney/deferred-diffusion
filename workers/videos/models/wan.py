import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
)
from huggingface_hub import hf_hub_download

from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_umt5_text_encoder,
)
from utils.utils import ensure_divisible, get_16_9_resolution, resize_image
from videos.context import VideoContext


@decorator_global_pipeline_cache
def get_pipeline_i2v(model_id, wan_2_1=False, torch_dtype=torch.bfloat16) -> WanImageToVideoPipeline:
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    transformer_2 = None
    if wan_2_1:
        transformer_2 = get_quantized_model(
            model_id=model_id,
            subfolder="transformer_2",
            model_class=WanTransformer3DModel,
            target_precision=4 if LOW_VRAM else 4,
            torch_dtype=torch_dtype,
        )

    text_encoder = get_quantized_umt5_text_encoder(8)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=text_encoder,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if LOW_VRAM or wan_2_1:
        # Wan T2V models are large, so use CPU offload if low VRAM
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


@decorator_global_pipeline_cache
def get_pipeline_t2v(model_id, wan_2_1=False, torch_dtype=torch.bfloat16) -> WanPipeline:
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    transformer_2 = None
    if wan_2_1:
        transformer_2 = get_quantized_model(
            model_id=model_id,
            subfolder="transformer_2",
            model_class=WanTransformer3DModel,
            target_precision=4 if LOW_VRAM else 4,
            torch_dtype=torch_dtype,
        )

    text_encoder = get_quantized_umt5_text_encoder(8)

    pipe = WanPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=text_encoder,
        vae=AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32),
        torch_dtype=torch_dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    if LOW_VRAM or wan_2_1:
        # Wan T2V models are large, so use CPU offload if low VRAM
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def text_to_video(context: VideoContext):

    if context.data.model == "wan-2-1":
        pipe = get_pipeline_t2v(model_id="magespace/Wan2.1-T2V-14B-Lightning-Diffusers", wan_2_1=False)
    else:
        pipe = get_pipeline_t2v(model_id="magespace/Wan2.2-T2V-A14B-Lightning-Diffusers", wan_2_1=True)

    width, height = get_16_9_resolution("480p")
    width = ensure_divisible(width, 16)
    height = ensure_divisible(height, 16)

    # Wan gives better results with a default negative prompt
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    output = pipe(
        width=width,
        height=height,
        prompt=context.data.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        # guidance_scale_2=1.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


def main(context: VideoContext):
    image = context.image
    if image is None:
        return text_to_video(context)

    if context.data.model == "wan-2-1":
        pipe = get_pipeline_i2v(model_id="magespace/Wan2.1-I2V-14B-480P-Lightning-Diffusers", wan_2_1=False)
    else:
        pipe = get_pipeline_i2v(model_id="magespace/Wan2.2-I2V-A14B-Lightning-Diffusers", wan_2_1=True)

    width, height = get_16_9_resolution("720p")
    image = resize_image(image, 16, 1.0, width, height)

    # Wan gives better results with a default negative prompt
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    output = pipe(
        width=image.size[0],
        height=image.size[1],
        image=image,
        prompt=context.data.prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        guidance_scale=1.0,
        # guidance_scale_2=1.0,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path

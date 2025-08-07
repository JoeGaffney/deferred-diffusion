import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from huggingface_hub import hf_hub_download

from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_umt5_text_encoder,
)
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext


# NOTE this one is heavy maybe we should not cache it globally
@decorator_global_pipeline_cache
def get_pipeline(model_id, torch_dtype=torch.bfloat16) -> WanImageToVideoPipeline:

    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    transformer_2 = get_quantized_model(
        model_id=model_id,
        subfolder="transformer_2",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch_dtype,
    )

    text_encoder = get_quantized_umt5_text_encoder(8)

    # NOTE adds more memory overhead
    # vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=transformer_2,
        text_encoder=text_encoder,
        # vae=vae,
        torch_dtype=torch_dtype,
    )
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)

    lighting_lora = False
    if lighting_lora:
        # NOTE this can give large speedups
        lora_path = hf_hub_download(
            repo_id="Kijai/WanVideo_comfy",
            filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
        )

        # Load LoRA weights directly
        pipe.load_lora_weights(lora_path)

        # NOTE shoudl we fuse for offloading?
        # pipe.fuse_lora()

    try:
        pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
        pipe.vae.enable_slicing()
    except:
        pass

    pipe.enable_model_cpu_offload()
    return pipe


def main(context: VideoContext):
    pipe = get_pipeline(model_id=context.data.model_path)
    image = context.image
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")

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
        # force to 1.0 as is way faster
        guidance_scale=1.0,  # context.data.guidance_scale,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=24)
    return processed_path

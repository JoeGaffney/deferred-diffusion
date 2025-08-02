import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from huggingface_hub import hf_hub_download
from transformers import UMT5EncoderModel

from common.memory import LOW_VRAM
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext

WAN_TRANSFORMER_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
UMT_T5_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


# NOTE this one is heavy maybe we should not cache it globally
@decorator_global_pipeline_cache
def get_pipeline(model_id, torch_dtype=torch.bfloat16) -> WanImageToVideoPipeline:

    # 2.1 use the same transformer for all variants
    transformer = get_quantized_model(
        model_id=WAN_TRANSFORMER_MODEL_PATH,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=4 if LOW_VRAM else 8,
        torch_dtype=torch_dtype,
    )

    text_encoder = get_quantized_model(
        model_id=UMT_T5_MODEL_PATH,
        subfolder="text_encoder",
        model_class=UMT5EncoderModel,
        target_precision=8,
        torch_dtype=torch_dtype,
    )

    # vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        transformer_2=None,
        text_encoder=text_encoder,
        # vae=vae,
        torch_dtype=torch_dtype,
    )

    # NOTE this can give large speedups
    # lora_path = hf_hub_download(
    #     repo_id="Kijai/WanVideo_comfy",
    #     filename="Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors",
    # )
    # pipe.load_lora_weights(lora_path)
    # 5b just has one transformer ???
    # pipe.model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"

    # NOTE there is choices around the schedulers
    # scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    # scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=4.0)

    # NOTE works with wan2.1 but not with wan2.2
    # try:
    #     pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
    #     pipe.vae.enable_slicing()
    # except:
    #     pass

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
        guidance_scale=context.data.guidance_scale,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path

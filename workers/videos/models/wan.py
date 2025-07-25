import torch
from diffusers import (
    AutoencoderKLWan,
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from transformers import CLIPVisionModel, UMT5EncoderModel

from common.memory import LOW_VRAM
from common.pipeline_helpers import get_quantized_model
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext

WAN_TRANSFORMER_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
UMT_T5_MODEL_PATH = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


# NOTE this one is heavy maybe we should not cache it globally
# @decorator_global_pipeline_cache
def get_pipeline(model_id, torch_dtype=torch.bfloat16):

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

    # NOTE this are recommended to be float32 for better quality but uses more memory
    # image_encoder = get_quantized_model(
    #     model_id=model_id,
    #     subfolder="image_encoder",
    #     model_class=CLIPVisionModel,
    #     target_precision=16,
    #     torch_dtype=torch.float32,
    # )

    # vae = get_quantized_model(
    #     model_id=model_id,
    #     subfolder="vae",
    #     model_class=AutoencoderKLWan,
    #     target_precision=16,
    #     torch_dtype=torch.float32,
    # )

    # Alt schedulers
    # NOTE stick with default as these can blur the video
    # scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    # scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
        # image_encoder=image_encoder,
        # vae=vae,
        # scheduler=scheduler,
    )

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

    output = pipe(
        width=image.size[0],
        height=image.size[1],
        image=image,
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        guidance_scale=context.data.guidance_scale,
        generator=context.get_generator(),
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path

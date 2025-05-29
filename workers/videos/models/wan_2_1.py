from venv import logger

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.hooks.group_offloading import apply_group_offloading
from transformers import CLIPVisionModel, UMT5EncoderModel

from common.pipeline_helpers import get_quantized_model
from utils.utils import (
    cache_info_decorator,
    get_16_9_resolution,
    resize_image,
    time_info_decorator,
)
from videos.context import VideoContext

torch.backends.cuda.matmul.allow_tf32 = True


@time_info_decorator
def get_pipeline(model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", torch_dtype=torch.bfloat16):
    image_encoder = get_quantized_model(
        model_id=model_id,
        subfolder="image_encoder",
        model_class=CLIPVisionModel,
        target_precision=16,
        torch_dtype=torch_dtype,
    )
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=WanTransformer3DModel,
        target_precision=8,
        torch_dtype=torch_dtype,
    )

    text_encoder = get_quantized_model(
        model_id=model_id,
        subfolder="text_encoder",
        model_class=UMT5EncoderModel,
        target_precision=8,
        torch_dtype=torch_dtype,
    )

    # Alt schedulers
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    # scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=4.0)

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        image_encoder=image_encoder,
        transformer=transformer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )

    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: VideoContext):
    pipe = get_pipeline()
    image = context.image
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")

    width, height = get_16_9_resolution("1080p")
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

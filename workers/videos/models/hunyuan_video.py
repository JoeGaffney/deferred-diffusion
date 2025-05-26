from functools import lru_cache

import torch
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
from transformers import CLIPTextModel

from common.logger import logger
from common.pipeline_helpers import get_quantized_model
from utils.utils import cache_info_decorator, get_16_9_resolution, resize_image
from videos.context import VideoContext


@cache_info_decorator
@lru_cache(maxsize=1)
def get_pipeline(model_id="hunyuanvideo-community/HunyuanVideo-I2V"):

    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=HunyuanVideoTransformer3DModel,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    # text_encoder = LlamaModel.from_pretrained(
    #     model_id, subfolder="text_encoder", quantization_config=quant_config, torch_dtype=torch.float16
    # )

    text_encoder_2 = get_quantized_model(
        model_id,
        subfolder="text_encoder_2",
        model_class=CLIPTextModel,
        target_precision=4,
        torch_dtype=torch.float16,
    )

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        # text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        torch_dtype=torch.float16,
    )

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def image_to_video(context: VideoContext):
    pipe = get_pipeline()
    width, height = get_16_9_resolution("1080p")
    image = context.image
    image = resize_image(image, 32, 1.0, width, height)

    video = pipe(
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

    processed_path = context.save_video(video, fps=15)
    return processed_path


def main(context: VideoContext):
    if context.image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    return image_to_video(context)

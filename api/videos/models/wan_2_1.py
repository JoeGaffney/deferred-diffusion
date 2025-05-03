import functools
import time
from functools import lru_cache
from venv import logger

import numpy as np
import torch
import torch.autograd.profiler as profiler
from diffusers import (
    AutoencoderKLWan,
    GGUFQuantizationConfig,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download
from transformers import (
    BitsAndBytesConfig,
    CLIPVisionModel,
    QuantoConfig,
    UMT5EncoderModel,
)

from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext
from videos.schemas import VideoRequest

quant_config = QuantoConfig(weights="int8")


# @lru_cache(maxsize=1)
def get_pipeline(model_id="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", torch_dtype=torch.float16):

    image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch_dtype)
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch_dtype)

    gguf_transformer_path = hf_hub_download(
        repo_id="city96/Wan2.1-I2V-14B-480P-gguf", filename="wan2.1-i2v-14b-480p-Q3_K_S.gguf"
    )
    transformer = WanTransformer3DModel.from_single_file(
        gguf_transformer_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        torch_dtype=torch_dtype,
    )

    text_encoder = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch_dtype,
        # quantization_config=quant_config,
    )

    # onload_device = torch.device("cuda")
    # offload_device = torch.device("cpu")

    # apply_group_offloading(
    #     text_encoder,
    #     onload_device=onload_device,
    #     offload_device=offload_device,
    #     offload_type="block_level",
    #     num_blocks_per_group=4,
    # )

    # transformer.enable_group_offload(
    #     onload_device=onload_device,
    #     offload_device=offload_device,
    #     offload_type="block_level",
    #     num_blocks_per_group=4,
    # )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    pipe.enable_model_cpu_offload()
    # Since we've offloaded the larger models alrady, we can move the rest of the model components to GPU
    # pipe.to("cuda")

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: VideoContext):
    pipe = get_pipeline()
    image = context.image
    if image is None:
        raise ValueError("Image not found. Please provide a valid image path.")

    width, height = get_16_9_resolution("540p")
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

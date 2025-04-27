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

from utils.utils import get_16_9_resolution
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

    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")

    apply_group_offloading(
        text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=4,
    )

    transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=4,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch_dtype,
    )
    # pipe.enable_model_cpu_offload()
    # Since we've offloaded the larger models alrady, we can move the rest of the model components to GPU
    pipe.to("cuda")

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: VideoContext):
    pipe = get_pipeline(context.model)
    # pipe.to("cuda")

    image_orig = context.load_image(division=16)
    image = load_image(
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    )
    # Ensure dimensions are divisible by 16
    width = (image_orig.width // 16) * 16
    height = (image_orig.height // 16) * 16
    print(f"Adjusted dimensions: width={width}, height={height}")

    image = image.resize((width, height))
    prompt = (
        "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
        "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    )
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=2,
        guidance_scale=2.0,
        num_inference_steps=2,
    ).frames[0]

    processed_path = context.save_video(output, fps=16)
    return processed_path


# NOTE this is still wip as invesigating the performance
if __name__ == "__main__":
    width, height = get_16_9_resolution("480p")
    # Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    main(
        VideoContext(
            VideoRequest(
                model=model_id,
                input_image_path="../tmp/tornado_v001.jpg",
                output_video_path="../tmp/output/wan_2_1.mp4",
                strength=0.2,
                prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
                num_inference_steps=50,
                max_width=864,
                max_height=480,
            )
        )
    )

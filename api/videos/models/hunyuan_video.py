from functools import lru_cache

import torch
from diffusers import (
    GGUFQuantizationConfig,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoTransformer3DModel,
)
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from transformers import BitsAndBytesConfig, CLIPTextModel, LlamaModel, QuantoConfig

from common.logger import logger
from utils.utils import get_16_9_resolution, resize_image
from videos.context import VideoContext
from videos.schemas import VideoRequest

# quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True) # won't offload to GPU
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
# quant_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
# quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16)


@lru_cache(maxsize=1)
def get_pipeline(model_id="hunyuanvideo-community/HunyuanVideo-I2V"):
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
    )
    # text_encoder = LlamaModel.from_pretrained(
    #     model_id, subfolder="text_encoder", quantization_config=quant_config, torch_dtype=torch.float16
    # )
    # text_encoder_2 = CLIPTextModel.from_pretrained(
    #     model_id, subfolder="text_encoder_2", quantization_config=quant_config, torch_dtype=torch.float16
    # )
    # gguf_transformer_path = hf_hub_download(
    #     repo_id="city96/HunyuanVideo-I2V-gguf", filename="hunyuan-video-i2v-720p-Q5_K_M.gguf"
    # )
    # transformer = HunyuanVideoTransformer3DModel.from_single_file(
    #     gguf_transformer_path,
    #     quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
    #     torch_dtype=torch.float16,
    # )

    pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.float16,
    )

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def image_to_video(context: VideoContext):
    pipe = get_pipeline()
    width, height = get_16_9_resolution("540p")
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

import math

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, T5EncoderModel

from common.logger import logger
from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_t5_text_encoder,
    optimize_pipeline,
)
from images.context import ImageContext, PipelineConfig


@decorator_global_pipeline_cache
def get_pipeline(config: PipelineConfig):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=config.model_id,
        subfolder="transformer",
        model_class=QwenImageTransformer2DModel,
        target_precision=16,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder"] = get_quantized_model(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        subfolder="",
        model_class=Qwen2_5_VLForConditionalGeneration,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    # From
    # https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),  # We use shift=3 in distillation
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),  # We use shift=3 in distillation
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,  # set shift_terminal to None
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    args["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    pipe = QwenImagePipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
    )
    return optimize_pipeline(pipe, offload=LOW_VRAM)


def text_to_image_call(context: ImageContext):
    pipe = get_pipeline(context.get_pipeline_config())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt + " Ultra HD, 4K, cinematic composition.",
        "negative_prompt": "",
        "num_inference_steps": 20,
        "generator": context.generator,
        # "guidance_scale": context.data.guidance_scale,
        "true_cfg_scale": 1.0,
    }

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    pipe = QwenImageImg2ImgPipeline.from_pipe(context.get_pipeline_config())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": context.color_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
    }

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = QwenImageImg2ImgPipeline.from_pipe(context.get_pipeline_config())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        # "negative_prompt": context.data.negative_prompt,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale * 10,  # range is from 1.5 to 100
        "strength": context.data.strength,
    }

    logger.info(f"Inpainting call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")

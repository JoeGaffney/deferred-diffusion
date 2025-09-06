import math

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPipeline,
    QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_qwen_2_5_text_encoder,
    optimize_pipeline,
)
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=QwenImageTransformer2DModel,
        target_precision=16,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder"] = get_quantized_qwen_2_5_text_encoder(4)

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
        model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
    )
    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


@decorator_global_pipeline_cache
def get_edit_pipeline(model_id):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=QwenImageTransformer2DModel,
        target_precision=16,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder"] = get_quantized_qwen_2_5_text_encoder(4)

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

    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit",
        torch_dtype=torch.bfloat16,
        **args,
    )
    pipe.load_lora_weights(
        "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
    )
    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


def text_to_image_call(context: ImageContext):
    pipe = get_pipeline("ovedrive/qwen-image-4bit")

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt + " Ultra HD, 4K, cinematic composition.",
        "negative_prompt": "",
        "num_inference_steps": 8,
        "generator": context.generator,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    pipe = QwenImageImg2ImgPipeline.from_pipe(get_pipeline("ovedrive/qwen-image-4bit"))

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": "",
        "image": context.color_image,
        "generator": context.generator,
        "strength": context.data.strength,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_edit_call(context: ImageContext):
    pipe = get_edit_pipeline("ovedrive/qwen-image-edit-4bit")

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": "",
        "image": context.color_image,
        "generator": context.generator,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = QwenImageInpaintPipeline.from_pipe(get_pipeline("ovedrive/qwen-image-4bit"))

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": "",
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_edit_call(context)
        # NOTE do we allow image to image with strength?
        # return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")

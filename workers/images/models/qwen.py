import math

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision
from PIL import Image

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from common.text_encoders import qwen_edit_encode, qwen_encode
from images.context import ImageContext

_use_nunchaku = True


def get_scheduler():
    # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
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
    return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


@decorator_global_pipeline_cache
def get_pipeline(model_id) -> QwenImagePipeline:
    if _use_nunchaku:
        rank = 128  # you can also use the rank=128 model to improve the quality
        model_paths = {
            4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
            8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
        }
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[8])
    else:
        transformer = get_quantized_model(
            model_id="ovedrive/qwen-image-4bit",
            subfolder="transformer",
            model_class=QwenImageTransformer2DModel,
            target_precision=16,
            torch_dtype=torch.bfloat16,
        )

    pipe = QwenImagePipeline.from_pretrained(
        model_id,
        text_encoder=None,
        tokenizer=None,
        scheduler=get_scheduler(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    if not _use_nunchaku:
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors"
        )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


@decorator_global_pipeline_cache
def get_edit_pipeline(model_id) -> QwenImageEditPlusPipeline:
    if _use_nunchaku:
        num_inference_steps = 8  # you can also use the 8-step model to improve the quality
        rank = 128  # you can also use the rank=128 model to improve the quality
        model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"
        transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)
    else:
        transformer = get_quantized_model(
            model_id="ovedrive/Qwen-Image-Edit-2509-4bit",
            subfolder="transformer",
            model_class=QwenImageTransformer2DModel,
            target_precision=16,
            torch_dtype=torch.bfloat16,
        )

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        text_encoder=None,
        tokenizer=None,
        transformer=transformer,
        scheduler=get_scheduler(),
        torch_dtype=torch.bfloat16,
    )
    if not _use_nunchaku:
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
        )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


def text_to_image_call(context: ImageContext):
    prompt_embeds, prompt_embeds_mask = qwen_encode(context.data.prompt + " Ultra HD, 4K, cinematic composition.")
    pipe = get_pipeline("Qwen/Qwen-Image")

    args = {
        "width": context.width,
        "height": context.height,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt": "",
        "num_inference_steps": 8,
        "generator": context.generator,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_edit_call(context: ImageContext):
    prompt_embeds, prompt_embeds_mask = qwen_edit_encode(
        context.data.prompt
    )  # + "Ultra HD, 4K, cinematic composition.")
    pipe = get_edit_pipeline("ovedrive/Qwen-Image-Edit-2509-4bit")

    # gather all possible reference images
    reference_images = []
    if context.color_image:
        reference_images.append(context.color_image)

    for current in context.get_reference_images():
        if current is not None:
            reference_images.append(current)

    args = {
        "width": context.width,
        "height": context.height,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt": "",
        "image": reference_images,
        "generator": context.generator,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    prompt_embeds, prompt_embeds_mask = qwen_encode(context.data.prompt + " Ultra HD, 4K, cinematic composition.")

    pipe = QwenImageInpaintPipeline.from_pipe(get_pipeline("ovedrive/qwen-image-4bit"))

    args = {
        "width": context.width,
        "height": context.height,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "negative_prompt": "",
        "image": context.color_image,
        "mask_image": context.mask_image,
        "generator": context.generator,
        "strength": context.data.strength,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    if context.color_image and context.mask_image:
        return inpainting_call(context)
    if context.color_image or context.get_reference_images() != []:
        return image_edit_call(context)
    return text_to_image_call(context)

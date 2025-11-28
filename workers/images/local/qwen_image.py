import math

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageInpaintPipeline,
    QwenImagePipeline,
)
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision
from PIL import Image

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from common.text_encoders import get_qwen2_5_text_encoder
from images.context import ImageContext


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
def get_pipeline(model_id, inpainting: bool) -> QwenImagePipeline | QwenImageInpaintPipeline:
    rank = 128  # you can also use the rank=128 model to improve the quality
    model_paths = {
        4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
        8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
    }
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[8])

    if inpainting:
        pipe = QwenImageInpaintPipeline.from_pretrained(
            model_id,
            text_encoder=get_qwen2_5_text_encoder(),
            scheduler=get_scheduler(),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = QwenImagePipeline.from_pretrained(
            model_id,
            text_encoder=get_qwen2_5_text_encoder(),
            scheduler=get_scheduler(),
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(23))


@decorator_global_pipeline_cache
def get_edit_pipeline(model_id) -> QwenImageEditPlusPipeline:
    num_inference_steps = 8  # you can also use the 8-step model to improve the quality
    rank = 128  # you can also use the rank=128 model to improve the quality
    model_path = f"nunchaku-tech/nunchaku-qwen-image-edit-2509/svdq-{get_precision()}_r{rank}-qwen-image-edit-2509-lightningv2.0-{num_inference_steps}steps.safetensors"
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_path)

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        text_encoder=get_qwen2_5_text_encoder(),
        transformer=transformer,
        scheduler=get_scheduler(),
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(23))


def text_to_image_call(context: ImageContext):
    pipe = get_pipeline("Qwen/Qwen-Image", inpainting=False)
    prompt = context.data.cleaned_prompt + " Ultra HD, 4K, cinematic composition."

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": prompt,
        "negative_prompt": " ",
        "num_inference_steps": 8,
        "generator": context.generator,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_edit_call(context: ImageContext):
    # see https://github.com/huggingface/diffusers/pull/12453/files
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_edit_module

    qwen_edit_module.VAE_IMAGE_SIZE = context.width * context.height

    # gather all possible reference images
    reference_images = []
    if context.color_image:
        reference_images.append(context.color_image)

    for current in context.get_reference_images():
        if current is not None:
            reference_images.append(current)

    # prompt_embeds, prompt_embeds_mask = qwen_edit_encode(context.data.cleaned_prompt, reference_images)
    pipe = get_edit_pipeline("Qwen/Qwen-Image-Edit-2509")

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.cleaned_prompt,
        "negative_prompt": " ",
        "image": reference_images,
        "generator": context.generator,
        "num_inference_steps": 8,
        "true_cfg_scale": 1.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_pipeline("Qwen/Qwen-Image", inpainting=True)
    prompt = context.data.cleaned_prompt + " Ultra HD, 4K, cinematic composition."

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": prompt,
        "negative_prompt": " ",
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

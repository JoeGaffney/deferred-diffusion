import math
from pathlib import Path
from typing import List

import torch

from common.monkey_patches import apply_qwen_image_patches

apply_qwen_image_patches()

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageInpaintPipeline,
    QwenImagePipeline,
)
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_precision

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    optimize_pipeline,
    task_log_callback,
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
def get_pipeline(model_id) -> QwenImagePipeline:
    rank = 128  # you can also use the rank=128 model to improve the quality
    model_paths = {
        4: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.0-4steps.safetensors",
        8: f"nunchaku-tech/nunchaku-qwen-image/svdq-{get_precision()}_r{rank}-qwen-image-lightningv1.1-8steps.safetensors",
    }
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(model_paths[8])

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
    pipe = get_pipeline("Qwen/Qwen-Image")
    prompt = context.data.cleaned_prompt + " Ultra HD, 4K, cinematic composition."

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=8,
        generator=context.generator,
        true_cfg_scale=1.0,
        callback_on_step_end=task_log_callback(8),  # type: ignore
    ).images[0]
    return [context.save_output(processed_image, index=0)]


def image_edit_call(context: ImageContext) -> List[Path]:
    # see https://github.com/huggingface/diffusers/pull/12453/files
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qwen_edit_module

    qwen_edit_module.VAE_IMAGE_SIZE = context.width * context.height
    prompt = context.data.cleaned_prompt

    # gather all possible reference images
    reference_images = []
    if context.mask_image and context.color_image:
        prompt = (
            "Use image 1 for the mask region for inpainting. And use image 2 for the base image only alter the mask region and aim for a seamless blend. "
            + prompt
        )
        reference_images.append(context.mask_image.convert("RGB"))
        reference_images.append(context.color_image)
    else:
        if context.color_image:
            reference_images.append(context.color_image)

        for current in context.get_reference_images():
            if current is not None:
                reference_images.append(current)

    pipe = get_edit_pipeline("Qwen/Qwen-Image-Edit-2509")

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=" ",
        image=reference_images,
        generator=context.generator,
        num_inference_steps=8,
        true_cfg_scale=1.0,
        callback_on_step_end=task_log_callback(8),  # type: ignore
    ).images[0]

    return [context.save_output(processed_image, index=0)]


def main(context: ImageContext) -> List[Path]:
    if context.color_image or context.get_reference_images() != []:
        return image_edit_call(context)
    return text_to_image_call(context)

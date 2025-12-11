import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DDIMScheduler,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    optimize_pipeline,
    task_log_callback,
)
from images.adapters import AdapterPipelineConfig
from images.context import ImageContext

_negative_prompt_default = "worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes"


@decorator_global_pipeline_cache
def get_pipeline(model_id, config: AdapterPipelineConfig) -> StableDiffusionXLPipeline:

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    if config.ip_adapter_models != ():
        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            subfolder=list(config.ip_adapter_subfolders),
            weight_name=list(config.ip_adapter_weights),
        )

        if config.ip_adapter_image_encoder_model != "":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.ip_adapter_image_encoder_model,
                subfolder=config.ip_adapter_image_encoder_subfolder,
                torch_dtype=torch.bfloat16,
            )
            pipe.image_encoder = image_encoder  # type: ignore
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return optimize_pipeline(pipe, offload=is_memory_exceeded(11))


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id, variant=None) -> StableDiffusionXLInpaintPipeline:
    args = {}
    if variant:
        args["variant"] = variant

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        **args,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(11))


def text_to_image_call(context: ImageContext):
    controlnets = context.control_nets.get_loaded_controlnets()
    pipe_args = {}
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForText2Image.from_pipe(
        get_pipeline("SG161222/RealVisXL_V4.0", context.adapters.get_adapter_pipeline_config()),
        requires_safety_checker=False,
        **pipe_args,
    )

    args = {}
    if context.control_nets.is_enabled():
        args["image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        pipe.set_ip_adapter_scale(context.adapters.get_scales())

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        num_inference_steps=35,
        generator=context.generator,
        guidance_scale=3.5,
        callback_on_step_end=task_log_callback(35),
        **args,
    ).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    controlnets = context.control_nets.get_loaded_controlnets()
    pipe_args = {}
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForImage2Image.from_pipe(
        get_pipeline("SG161222/RealVisXL_V4.0", context.adapters.get_adapter_pipeline_config()),
        requires_safety_checker=False,
        **pipe_args,
    )

    args = {}
    if context.control_nets.is_enabled():
        args["control_image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        pipe.set_ip_adapter_scale(context.adapters.get_scales())

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        image=context.color_image,
        num_inference_steps=35,
        generator=context.generator,
        strength=context.data.strength,
        guidance_scale=3.5,
        callback_on_step_end=task_log_callback(35),
        **args,
    ).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_inpainting_pipeline("OzzyGT/RealVisXL_V4.0_inpainting", variant="fp16")

    processed_image = pipe.__call__(
        width=context.width,
        height=context.height,
        prompt=context.data.cleaned_prompt,
        negative_prompt=_negative_prompt_default,
        image=context.color_image,  # type: ignore
        mask_image=context.mask_image,  # type: ignore
        num_inference_steps=35,
        generator=context.generator,
        strength=0.95,
        guidance_scale=4.0,
        callback_on_step_end=task_log_callback(35),  # type: ignore
    ).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    context.ensure_divisible(16)

    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)

    return text_to_image_call(context)

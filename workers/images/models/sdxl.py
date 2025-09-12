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

from common.config import IMAGE_CPU_OFFLOAD
from common.pipeline_helpers import decorator_global_pipeline_cache, optimize_pipeline
from images.adapters import AdapterPipelineConfig
from images.context import ImageContext

_negative_prompt_default = "worst quality, inconsistent motion, blurry, jittery, distorted, render, cartoon, 3d, lowres, fused fingers, face asymmetry, eyes asymmetry, deformed eyes"


@decorator_global_pipeline_cache
def get_pipeline(model_id, config: AdapterPipelineConfig) -> StableDiffusionXLPipeline:

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
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
                torch_dtype=torch.float16,
            )
            pipe.image_encoder = image_encoder
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id, variant=None) -> StableDiffusionXLInpaintPipeline:
    args = {}
    if variant:
        args["variant"] = variant

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        **args,
    )

    return optimize_pipeline(pipe, offload=False)


def setup_controlnets_and_ip_adapters(pipe, context: ImageContext, args):
    if context.control_nets.is_enabled():
        args["image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        args["cross_attention_kwargs"] = {"ip_adapter_masks": context.adapters.get_masks()}
        pipe = context.adapters.set_scale(pipe)

    return pipe, args


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

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": _negative_prompt_default,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 3.5,
    }
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
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

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": _negative_prompt_default,
        "image": context.color_image,
        "num_inference_steps": 30,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": 3.5,
    }

    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_inpainting_pipeline("OzzyGT/RealVisXL_V4.0_inpainting", variant="fp16")

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        # "negative_prompt": _negative_prompt_default,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": 30,
        "generator": context.generator,
        "strength": 0.99,
        "guidance_scale": 5.0,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    context.ensure_divisible(16)
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")

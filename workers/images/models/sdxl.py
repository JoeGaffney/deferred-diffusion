import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image
from transformers import CLIPVisionModelWithProjection

from common.logger import logger
from common.pipeline_helpers import decorator_global_pipeline_cache, optimize_pipeline
from images.context import ImageContext, PipelineConfig


@decorator_global_pipeline_cache
def get_pipeline(config: PipelineConfig):
    args = {"torch_dtype": config.torch_dtype, "use_safetensors": True}

    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        **args,
    )

    if config.ip_adapter_models != ():
        if not hasattr(pipe, "load_ip_adapter"):
            raise ValueError("The pipeline does not support IP-Adapters. Please use a compatible pipeline.")

        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            subfolder=list(config.ip_adapter_subfolders),
            weight_name=list(config.ip_adapter_weights),
        )

        if config.ip_adapter_image_encoder_model != "":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.ip_adapter_image_encoder_model,
                subfolder=config.ip_adapter_image_encoder_subfolder,
                torch_dtype=config.torch_dtype,
            )
            pipe.image_encoder = image_encoder
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return optimize_pipeline(pipe, sequential_cpu_offload=False)


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
    def get_text_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForText2Image.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_text_pipeline(context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    def get_image_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForImage2Image.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_image_pipeline(context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets())

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

    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    def get_inpainting_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
        args = {}
        if controlnets != []:
            args["controlnet"] = controlnets

        return AutoPipelineForInpainting.from_pipe(
            get_pipeline(pipeline_config), requires_safety_checker=False, **args
        )

    pipe = get_inpainting_pipeline(
        context.get_pipeline_config(), controlnets=context.control_nets.get_loaded_controlnets()
    )
    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
    }
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

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

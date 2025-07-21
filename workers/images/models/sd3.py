import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from PIL import Image
from transformers import T5EncoderModel

from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from images.context import ImageContext, PipelineConfig

T5_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"


@decorator_global_pipeline_cache
def get_pipeline(config: PipelineConfig):
    args = {"torch_dtype": torch.bfloat16, "use_safetensors": True}
    args["transformer"] = get_quantized_model(
        model_id=config.model_id,
        subfolder="transformer",
        model_class=SD3Transformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder_3"] = get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",  # this is correct as we are loading the T5 encoder from the Flux model
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        config.model_id,
        **args,
    )

    return optimize_pipeline(pipe)


def setup_controlnets_and_ip_adapters(pipe, context: ImageContext, args):
    if context.control_nets.is_enabled():
        args["control_image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

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
    mode = "img_to_img"
    if context.data.mask:
        mode = "img_to_img_inpainting"
    if context.data.image is None:
        mode = "text_to_image"

    # SD3 workaround for controlnets
    if context.control_nets.is_enabled():
        return text_to_image_call(context)

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")

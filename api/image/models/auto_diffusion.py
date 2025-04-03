from functools import lru_cache

from common.pipeline_helpers import optimize_pipeline
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DiffusionPipeline,
    StableDiffusion3ControlNetPipeline,
)
from image.context import ImageContext
from image.models.diffusers_helpers import (
    image_to_image_call,
    inpainting_call,
    text_to_image_call,
)
from image.schemas import PipelineConfig
from transformers import CLIPVisionModelWithProjection
from utils.logger import logger
from utils.utils import cache_info_decorator


@cache_info_decorator
@lru_cache(maxsize=4)  # Cache up to 4 different pipelines
def get_pipeline(config: PipelineConfig):
    args = {"torch_dtype": config.torch_dtype, "use_safetensors": True}

    # this can really eat up the memory
    if config.disable_text_encoder_3 == True:
        args["text_encoder_3"] = None
        args["tokenizer_3"] = None

    pipe = DiffusionPipeline.from_pretrained(
        config.model_id,
        **args,
    )
    if config.ip_adapter_models != ():
        if not hasattr(pipe, "load_ip_adapter"):
            raise ValueError("The pipeline does not support IP-Adapters. Please use a compatible pipeline.")

        # load multiple as arrays
        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            subfolder=list(config.ip_adapter_subfolders),
            weight_name=list(config.ip_adapter_weights),
        )

        # image_encoder is require for some adapters
        if config.ip_adapter_image_encoder_model != "":
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                config.ip_adapter_image_encoder_model,
                subfolder=config.ip_adapter_image_encoder_subfolder,
                torch_dtype=config.torch_dtype,
            )
            pipe.image_encoder = image_encoder

    return optimize_pipeline(pipe)


def get_text_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets

    return AutoPipelineForText2Image.from_pipe(get_pipeline(pipeline_config), requires_safety_checker=False, **args)


def get_image_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets

    return AutoPipelineForImage2Image.from_pipe(get_pipeline(pipeline_config), requires_safety_checker=False, **args)


def get_inpainting_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets
    return AutoPipelineForInpainting.from_pipe(get_pipeline(pipeline_config), requires_safety_checker=False, **args)


# need to grab direct as SD3 control nets not full supported by diffusers
def get_sd3_controlnet_pipeline(pipeline_config: PipelineConfig, controlnets=[]):
    pipe = StableDiffusion3ControlNetPipeline.from_pipe(get_pipeline(pipeline_config), controlnet=controlnets)

    logger.warning(f"Loaded pipeline {pipeline_config.model_id} with controlnets")
    return pipe


# work around as SD3 control nets not full supported by diffusers
def main_sd3_controlnets(
    context: ImageContext, model_id="stabilityai/stable-diffusion-3-medium-diffusers", mode="text"
):
    controlnets = context.get_loaded_controlnets()
    pipeline_config = context.get_pipeline_config()

    # work around as SD3 not full supported by diffusers
    # there is no dedicated img to img or inpainting control net for SD3 atm
    return text_to_image_call(
        get_sd3_controlnet_pipeline(pipeline_config, controlnets=controlnets),
        context,
    )


def main(
    context: ImageContext,
    mode="text",
):
    if context.sd3_controlnet_mode == True:
        return main_sd3_controlnets(context, model_id=context.model, mode=mode)

    controlnets = context.get_loaded_controlnets()
    pipeline_config = context.get_pipeline_config()

    if mode == "text_to_image":
        return text_to_image_call(get_text_pipeline(pipeline_config, controlnets=controlnets), context)
    elif mode == "img_to_img":
        return image_to_image_call(get_image_pipeline(pipeline_config, controlnets=controlnets), context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(get_inpainting_pipeline(pipeline_config, controlnets=controlnets), context)

    return "invalid mode"

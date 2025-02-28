from functools import lru_cache

import torch
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
from utils.logger import logger


@lru_cache(maxsize=4)  # Cache up to 4 different pipelines
def get_pipeline(model_id, torch_dtype=torch.float16, disable_text_encoder_3=True):

    args = {"torch_dtype": torch_dtype, "use_safetensors": True}

    # this can really eat up the memory
    if disable_text_encoder_3 == True:
        args["text_encoder_3"] = None
        args["tokenizer_3"] = None

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        **args,
    )

    logger.warning(f"Loaded pipeline {model_id}")
    return optimize_pipeline(pipe)


def get_text_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets

    return AutoPipelineForText2Image.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
        **args,
    )


def get_image_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets

    return AutoPipelineForImage2Image.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
        **args,
    )


def get_inpainting_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    args = {}
    if controlnets != []:
        args["controlnet"] = controlnets
    return AutoPipelineForInpainting.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        requires_safety_checker=False,
        **args,
    )


# need to grab direct as SD3 control nets not full supported by diffusers
def get_sd3_controlnet_pipeline(model_id, torch_dtype=torch.float16, controlnets=[], disable_text_encoder_3=True):
    pipe = StableDiffusion3ControlNetPipeline.from_pipe(
        get_pipeline(model_id, torch_dtype=torch_dtype, disable_text_encoder_3=disable_text_encoder_3),
        controlnet=controlnets,
    )

    logger.warning(f"Loaded pipeline {model_id} with controlnets")
    return pipe


# work around as SD3 control nets not full supported by diffusers
def main_sd3_controlnets(
    context: ImageContext, model_id="stabilityai/stable-diffusion-3-medium-diffusers", mode="text"
):
    disable_text_encoder_3 = context.disable_text_encoder_3
    controlnets = context.get_loaded_controlnets()
    torch_dtype = context.torch_dtype

    # work around as SD3 not full supported by diffusers
    # there is no dedicated img to img or inpainting control net for SD3 atm
    return text_to_image_call(
        get_sd3_controlnet_pipeline(
            model_id,
            torch_dtype=torch_dtype,
            controlnets=controlnets,
            disable_text_encoder_3=disable_text_encoder_3,
        ),
        context,
    )


def main(
    context: ImageContext,
    mode="text",
):
    if context.sd3_controlnet_mode == True:
        return main_sd3_controlnets(context, model_id=context.model, mode=mode)

    disable_text_encoder_3 = context.disable_text_encoder_3
    controlnets = context.get_loaded_controlnets()
    torch_dtype = context.torch_dtype

    if mode == "text_to_image":
        return text_to_image_call(
            get_text_pipeline(
                context.model,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )
    elif mode == "img_to_img":
        return image_to_image_call(
            get_image_pipeline(
                context.model,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )
    elif mode == "img_to_img_inpainting":
        return inpainting_call(
            get_inpainting_pipeline(
                context.model,
                torch_dtype=torch_dtype,
                controlnets=controlnets,
                disable_text_encoder_3=disable_text_encoder_3,
            ),
            context,
        )

    return "invalid mode"

from functools import lru_cache

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    DDIMScheduler,
    DiffusionPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    GGUFQuantizationConfig,
)
from transformers import (
    BitsAndBytesConfig,
    CLIPVisionModelWithProjection,
    QuantoConfig,
    T5EncoderModel,
)

from common.pipeline_helpers import optimize_pipeline
from image.context import ImageContext
from image.models.diffusers_helpers import (
    image_to_image_call,
    inpainting_call,
    text_to_image_call,
)
from image.schemas import PipelineConfig
from utils.utils import cache_info_decorator

quant_config = QuantoConfig(weights="int8")


def get_pipeline_flux(config: PipelineConfig):

    transformer = FluxTransformer2DModel.from_single_file(
        config.model_transformer_guf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
    )

    # NOTE invesigate quantization config
    pipe = FluxPipeline.from_pretrained(
        config.model_id,
        transformer=transformer,
        # text_encoder_2=T5EncoderModel.from_pretrained(
        #     config.model_id,
        #     subfolder="text_encoder_2",
        #     quantization_config=quant_config,
        #     torch_dtype=torch.bfloat16,
        # ),
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, sequential_cpu_offload=config.optimize_low_vram)


@cache_info_decorator
@lru_cache(maxsize=4)  # Cache up to 4 different pipelines
def get_pipeline(config: PipelineConfig):
    if config.model_family == "flux":
        return get_pipeline_flux(config)

    args = {"torch_dtype": config.torch_dtype, "use_safetensors": True}

    # this can really eat up the memory
    if config.model_family == "sd3":
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

            # Supposed to help with consistency
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return optimize_pipeline(pipe, sequential_cpu_offload=config.optimize_low_vram)


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


def main(
    context: ImageContext,
    mode="text",
):
    controlnets = context.get_loaded_controlnets()
    pipeline_config = context.get_pipeline_config()

    # work around as SD3 not full supported by diffusers
    if context.controlnets_enabled == True and pipeline_config.model_family == "sd3":
        return text_to_image_call(get_text_pipeline(pipeline_config, controlnets=controlnets), context)

    if mode == "text_to_image":
        return text_to_image_call(get_text_pipeline(pipeline_config, controlnets=controlnets), context)
    elif mode == "img_to_img":
        return image_to_image_call(get_image_pipeline(pipeline_config, controlnets=controlnets), context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(get_inpainting_pipeline(pipeline_config, controlnets=controlnets), context)

    return "invalid mode"

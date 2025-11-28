import torch
from diffusers import (
    AutoPipelineForText2Image,
    FluxFillPipeline,
    FluxKontextPipeline,
    FluxPipeline,
)
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision
from PIL import Image

from common.memory import is_memory_exceeded
from common.pipeline_helpers import decorator_global_pipeline_cache, optimize_pipeline
from common.text_encoders import get_t5_text_encoder
from images.adapters import AdapterPipelineConfig
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id, config: AdapterPipelineConfig):
    # Controlnet is not supported for FluxTransformer2DModelV2 for now
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{get_precision()}_r32-flux.1-krea-dev.safetensors"
    )

    pipe = FluxPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    if config.ip_adapter_models != ():
        if not hasattr(pipe, "load_ip_adapter"):
            raise ValueError("The pipeline does not support IP-Adapters. Please use a compatible pipeline.")

        pipe.load_ip_adapter(
            list(config.ip_adapter_models),
            weight_name=list(config.ip_adapter_weights),
            image_encoder_pretrained_model_name_or_path=config.ip_adapter_image_encoder_subfolder,
        )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


@decorator_global_pipeline_cache
def get_kontext_pipeline(model_id):
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
    )

    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id):
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors"
    )

    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        text_encoder_2=get_t5_text_encoder(),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


def text_to_image_call(context: ImageContext):
    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForText2Image.from_pipe(
        get_pipeline("black-forest-labs/FLUX.1-Krea-dev", context.adapters.get_adapter_pipeline_config()),
        requires_safety_checker=False,
        **pipe_args,
    )

    args = {
        "prompt": context.data.cleaned_prompt,
        "width": context.width,
        "height": context.height,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 2.5,
    }
    if context.control_nets.is_enabled():
        args["control_image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        pipe.set_ip_adapter_scale(context.adapters.get_scales())

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()
    return processed_image


def image_to_image_call(context: ImageContext):
    pipe = get_kontext_pipeline("black-forest-labs/FLUX.1-Kontext-dev")

    args = {
        "prompt": context.data.cleaned_prompt,
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 2.0,
    }
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_inpainting_pipeline("black-forest-labs/FLUX.1-Fill-dev")

    args = {
        "prompt": context.data.cleaned_prompt,
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 30,
        "strength": context.data.strength,
    }

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    if context.color_image and context.mask_image:
        return inpainting_call(context)
    elif context.color_image:
        return image_to_image_call(context)
    return text_to_image_call(context)

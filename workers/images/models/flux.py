import torch
from diffusers import (
    AutoPipelineForText2Image,
    FluxFillPipeline,
    FluxKontextPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from nunchaku import NunchakuFluxTransformer2dModel, NunchakuFluxTransformer2DModelV2
from nunchaku.utils import get_precision
from PIL import Image

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from common.text_encoders import flux_encode
from images.adapters import AdapterPipelineConfig
from images.context import ImageContext

_use_nunchaku = True


@decorator_global_pipeline_cache
def get_pipeline(model_id, config: AdapterPipelineConfig):
    if _use_nunchaku:
        # Controlnet is not supported for FluxTransformer2DModelV2 for now
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-krea-dev/svdq-{get_precision()}_r32-flux.1-krea-dev.safetensors"
        )
    else:
        transformer = get_quantized_model(
            model_id=model_id,
            subfolder="transformer",
            model_class=FluxTransformer2DModel,
            target_precision=IMAGE_TRANSFORMER_PRECISION,
            torch_dtype=torch.bfloat16,
        )

    pipe = FluxPipeline.from_pretrained(
        model_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
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

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


@decorator_global_pipeline_cache
def get_kontext_pipeline(model_id):
    if _use_nunchaku:
        transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-kontext-dev/svdq-{get_precision()}_r32-flux.1-kontext-dev.safetensors"
        )
    else:
        transformer = get_quantized_model(
            model_id=model_id,
            subfolder="transformer",
            model_class=FluxTransformer2DModel,
            target_precision=IMAGE_TRANSFORMER_PRECISION,
            torch_dtype=torch.bfloat16,
        )

    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


@decorator_global_pipeline_cache
def get_inpainting_pipeline(model_id):
    if _use_nunchaku:
        transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"nunchaku-tech/nunchaku-flux.1-fill-dev/svdq-{get_precision()}_r32-flux.1-fill-dev.safetensors"
        )
    else:
        transformer = get_quantized_model(
            model_id=model_id,
            subfolder="transformer",
            model_class=FluxTransformer2DModel,
            target_precision=IMAGE_TRANSFORMER_PRECISION,
            torch_dtype=torch.bfloat16,
        )

    pipe = FluxFillPipeline.from_pretrained(
        model_id,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


def apply_prompt_embeddings(args, prompt, negative_prompt=""):
    prompt_embeds, pooled_prompt_embeds = flux_encode(prompt)
    args["prompt_embeds"] = prompt_embeds
    args["pooled_prompt_embeds"] = pooled_prompt_embeds

    if negative_prompt != "":
        negative_prompt_embeds, negative_pooled_prompt_embeds = flux_encode(negative_prompt)
        args["negative_prompt_embeds"] = negative_prompt_embeds
        args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

    return args


def text_to_image_call(context: ImageContext):
    # NOTE just use krea for now as it seems to be better
    # model_id = "black-forest-labs/FLUX.1-dev"
    model_id = "black-forest-labs/FLUX.1-Krea-dev"

    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForText2Image.from_pipe(
        get_pipeline(model_id, context.adapters.get_adapter_pipeline_config()),
        requires_safety_checker=False,
        **pipe_args,
    )

    args = {
        "width": context.width,
        "height": context.height,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 2.5,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
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
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "num_inference_steps": 20,
        "generator": context.generator,
        "guidance_scale": 2.0,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_inpainting_pipeline("black-forest-labs/FLUX.1-Fill-dev")

    args = {
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": 30,
        "generator": context.generator,
        "guidance_scale": 30,
        "strength": context.data.strength,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")

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

import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    FluxFillPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from common.text_encoders import get_pipeline_flux_text_encoder
from images.adapters import AdapterPipelineConfig
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id, config: AdapterPipelineConfig):
    args = {}

    args["transformer"] = get_quantized_model(
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
        torch_dtype=torch.bfloat16,
        **args,
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
def get_inpainting_pipeline(model_id):
    args = {}

    args["transformer"] = get_quantized_model(
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
        torch_dtype=torch.bfloat16,
        **args,
    )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


def apply_prompt_embeddings(args, prompt, negative_prompt=""):
    pipe = get_pipeline_flux_text_encoder()
    prompt_embeds, pooled_prompt_embeds = pipe.encode(prompt)
    args["prompt_embeds"] = prompt_embeds
    args["pooled_prompt_embeds"] = pooled_prompt_embeds

    if negative_prompt != "":
        negative_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode(negative_prompt)
        args["negative_prompt_embeds"] = negative_prompt_embeds
        args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

    return args


def setup_controlnets_and_ip_adapters(pipe, context: ImageContext, args):
    if context.control_nets.is_enabled():
        args["control_image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    if context.adapters.is_enabled():
        args["ip_adapter_image"] = context.adapters.get_images()
        pipe = context.adapters.set_scale(pipe)

    return pipe, args


def text_to_image_call(context: ImageContext, model_id):
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
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()
    return processed_image


def image_to_image_call(context: ImageContext, model_id):
    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForImage2Image.from_pipe(
        get_pipeline(model_id, context.adapters.get_adapter_pipeline_config()),
        requires_safety_checker=False,
        **pipe_args,
    )

    args = {
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext, model_id):
    pipe = get_inpainting_pipeline(model_id)

    args = {
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "mask_image": context.mask_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale * 10,  # range is from 1.5 to 100
        "strength": context.data.strength,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    model_id = "black-forest-labs/FLUX.1-dev"
    if context.data.model == "flux-1-krea":
        model_id = "black-forest-labs/FLUX.1-Krea-dev"

    if mode == "text_to_image":
        return text_to_image_call(context, model_id=model_id)
    elif mode == "img_to_img":
        return image_to_image_call(context, model_id=model_id)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context, model_id="black-forest-labs/FLUX.1-Fill-dev")

    raise ValueError(f"Unknown mode: {mode}")

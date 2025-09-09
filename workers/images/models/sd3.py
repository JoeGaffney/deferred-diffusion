import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from PIL import Image

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from common.text_encoders import sd3_encode
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    transformer = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=SD3Transformer2DModel,
        target_precision=IMAGE_TRANSFORMER_PRECISION,
        torch_dtype=torch.bfloat16,
    )

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    return optimize_pipeline(pipe, offload=IMAGE_CPU_OFFLOAD)


def apply_prompt_embeddings(args, prompt, negative_prompt=""):
    prompt_embeds, pooled_prompt_embeds = sd3_encode(prompt)
    args["prompt_embeds"] = prompt_embeds
    args["pooled_prompt_embeds"] = pooled_prompt_embeds

    if negative_prompt != "":
        negative_prompt_embeds, negative_pooled_prompt_embeds = sd3_encode(negative_prompt)
        args["negative_prompt_embeds"] = negative_prompt_embeds
        args["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

    return args


def setup_controlnets_and_ip_adapters(pipe, context: ImageContext, args):
    if context.control_nets.is_enabled():
        args["control_image"] = context.control_nets.get_images()
        args["controlnet_conditioning_scale"] = context.control_nets.get_conditioning_scales()

    return pipe, args


def text_to_image_call(context: ImageContext):
    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForText2Image.from_pipe(
        get_pipeline("stabilityai/stable-diffusion-3.5-large"), requires_safety_checker=False, **pipe_args
    )

    args = {
        "width": context.width,
        "height": context.height,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, context.data.negative_prompt)
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForImage2Image.from_pipe(
        get_pipeline("stabilityai/stable-diffusion-3.5-large"), requires_safety_checker=False, **pipe_args
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
    args = apply_prompt_embeddings(args, context.data.prompt, context.data.negative_prompt)
    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe_args = {}
    controlnets = context.control_nets.get_loaded_controlnets()
    if controlnets != []:
        pipe_args["controlnet"] = controlnets

    pipe = AutoPipelineForInpainting.from_pipe(
        get_pipeline("stabilityai/stable-diffusion-3.5-large"), requires_safety_checker=False, **pipe_args
    )

    args = {
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "mask_image": context.mask_image,
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


def main(context: ImageContext) -> Image.Image:
    context.ensure_divisible(16)

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

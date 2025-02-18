import torch
from transformers import T5EncoderModel, BitsAndBytesConfig

from utils.utils import get_16_9_resolution
from common.context import Context


def text_to_image_call(pipe, context: Context, use_image_wh=False):
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()
    wh = context.resize_max_wh(division=16)
    if use_image_wh:
        image = context.load_image(division=16)  # Load input image
        wh = image.size

    args = {
        "width": wh[0],
        "height": wh[1],
        "prompt": context.prompt,
        "negative_prompt": context.negative_prompt,
        "num_inference_steps": context.num_inference_steps,
        "generator": generator,
        "guidance_scale": context.guidance_scale,
    }
    if context.controlnets_enabled:
        # different pattern of arguments
        if context.sd3_controlnet_mode:
            args["control_image"] = context.get_controlnet_images()
        else:
            args["image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    processed_image = pipe.__call__(**args).images[0]

    if use_image_wh:
        processed_image = context.resize_image_to_orig(processed_image)
    else:
        processed_image = context.resize_image_to_max_wh(processed_image)

    processed_path = context.save_image(processed_image)
    return processed_path


def image_to_image_call(pipe, context: Context):
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()

    args = {
        "width": image.size[0],
        "height": image.size[1],
        "prompt": context.prompt,
        "negative_prompt": context.negative_prompt,
        "image": image,
        "num_inference_steps": context.num_inference_steps,
        "generator": generator,
        "strength": context.strength,
        "guidance_scale": context.guidance_scale,
    }
    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    processed_image = pipe.__call__(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def inpainting_call(pipe, context: Context):
    image = context.load_image(division=16)
    mask = context.load_mask()
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()

    args = {
        "width": image.size[0],
        "height": image.size[1],
        "prompt": context.prompt,
        "negative_prompt": context.negative_prompt,
        "image": image,
        "mask_image": mask,
        "num_inference_steps": context.num_inference_steps,
        "generator": generator,
        "strength": context.strength,
        "guidance_scale": context.guidance_scale,
        "padding_mask_crop": None if context.inpainting_full_image == True else 32,
    }
    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    processed_image = pipe(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def upscale_call(pipe, context: Context):

    width, height = get_16_9_resolution("540p")  # 4k
    # width, height = get_16_9_resolution("360p")  # 1440p
    context.max_width = width
    context.max_height = height

    image = context.load_image(division=16, scale=0.5)
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    processed_image = pipe(
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        guidance_scale=context.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image, scale=2)
    processed_path = context.save_image(processed_image)
    return processed_path


def optimize_pipeline(pipe, disable_safety_checker=True):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
    pipe.enable_attention_slicing("auto")  # Enable attention slicing for faster inference
    if disable_safety_checker:
        pipe.safety_checker = dummy_safety_checker

    return pipe


quantization_config = BitsAndBytesConfig(load_in_8bit=True)


def get_t5_quantized(model_id):

    return T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_3",
        quantization_config=quantization_config,
    )

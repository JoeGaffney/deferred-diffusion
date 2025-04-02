from image.context import ImageContext
from utils.logger import logger


def text_to_image_call(pipe, context: ImageContext):

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }

    if context.controlnets_enabled:
        # different pattern of arguments
        if context.sd3_controlnet_mode:
            args["control_image"] = context.get_controlnet_images()
        else:
            args["image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    if context.ip_adapters_enabled:
        args["ip_adapter_image"] = context.get_ip_adapter_images()
        pipe = context.load_ip_adapter(pipe)
    else:
        context.unload_ip_adapter(pipe)

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    processed_image = context.resize_image_to_orig(processed_image)

    processed_path = context.save_image(processed_image)
    return processed_path


def image_to_image_call(pipe, context: ImageContext):

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

    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    if context.ip_adapters_enabled:
        args["ip_adapter_image"] = context.get_ip_adapter_images()
        pipe = context.load_ip_adapter(pipe)
    else:
        context.unload_ip_adapter(pipe)

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def inpainting_call(pipe, context: ImageContext):
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
        "padding_mask_crop": None if context.data.inpainting_full_image == True else 32,
    }

    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images()
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    if context.ip_adapters_enabled:
        args["ip_adapter_image"] = context.get_ip_adapter_images()
        pipe = context.load_ip_adapter(pipe)
    else:
        context.unload_ip_adapter(pipe)

    logger.info(f"Inpainting call {args}")
    processed_image = pipe(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def upscale_call(pipe, context: ImageContext, scale=4):

    processed_image = pipe(
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        image=context.color_image,
        num_inference_steps=context.data.num_inference_steps,
        generator=context.generator,
        guidance_scale=context.data.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image, scale=scale)
    processed_path = context.save_image(processed_image)
    return processed_path

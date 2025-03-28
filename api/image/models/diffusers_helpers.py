import torch
from image.context import ImageContext
from utils.logger import logger


def text_to_image_call(pipe, context: ImageContext):
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)

    # is there is base image attached use it's dimensions
    use_image_wh = False
    if context.data.input_image_path != "":
        use_image_wh = True

    wh = context.resize_max_wh(division=16)
    if use_image_wh:
        image = context.load_image(division=16)  # Load input image
        wh = image.size

    args = {
        "width": wh[0],
        "height": wh[1],
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": generator,
        "guidance_scale": context.data.guidance_scale,
    }
    if context.controlnets_enabled:
        # different pattern of arguments
        if context.sd3_controlnet_mode:
            args["control_image"] = context.get_controlnet_images(wh)
        else:
            args["image"] = context.get_controlnet_images(wh)
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]

    if use_image_wh:
        processed_image = context.resize_image_to_orig(processed_image)
    else:
        processed_image = context.resize_image_to_max_wh(processed_image)

    processed_path = context.save_image(processed_image)
    return processed_path


def image_to_image_call(pipe, context: ImageContext):
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)

    args = {
        "width": image.size[0],
        "height": image.size[1],
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
    }
    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images(image.size)
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def inpainting_call(pipe, context: ImageContext):
    image = context.load_image(division=16)
    mask = context.load_mask(image.size)
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)

    args = {
        "width": image.size[0],
        "height": image.size[1],
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "image": image,
        "mask_image": mask,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": generator,
        "strength": context.data.strength,
        "guidance_scale": context.data.guidance_scale,
        "padding_mask_crop": None if context.data.inpainting_full_image == True else 32,
    }
    if context.controlnets_enabled:
        args["control_image"] = context.get_controlnet_images(image.size)
        args["controlnet_conditioning_scale"] = context.get_controlnet_conditioning_scales()

    logger.info(f"Inpainting call {args}")
    processed_image = pipe(**args).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def upscale_call(pipe, context: ImageContext, scale=4):
    image = context.load_image(division=16)
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)

    processed_image = pipe(
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        image=image,
        num_inference_steps=context.data.num_inference_steps,
        generator=generator,
        guidance_scale=context.data.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image, scale=scale)
    processed_path = context.save_image(processed_image)
    return processed_path

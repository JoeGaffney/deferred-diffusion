import torch

from common.context import Context


def diffusers_call(pipe, context: Context):
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()

    processed_image = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
        guidance_scale=context.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def diffusers_image_call(pipe, context: Context):
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()

    processed_image = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
        guidance_scale=context.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def diffusers_controlnet_call(pipe, context: Context):
    image = context.load_image(division=16)  # Load input image
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()

    processed_image = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        control_image=image,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        controlnet_conditioning_scale=context.strength,
        guidance_scale=context.guidance_scale,
        # max_sequence_length=77,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path


def diffusers_inpainting_call(pipe, context: Context):
    image = context.load_image()
    mask = context.load_mask()
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    processed_image = pipe(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        strength=context.strength,
        guidance_scale=context.guidance_scale,
        padding_mask_crop=32,
    ).images[0]

    processed_image = context.resize_image_to_orig(processed_image)
    processed_path = context.save_image(processed_image)
    return processed_path

import torch
from transformers import T5EncoderModel, BitsAndBytesConfig

from utils.utils import get_16_9_resolution
from common.context import Context


def diffusers_call(pipe, context: Context):
    generator = torch.Generator(device="cuda").manual_seed(context.seed)
    context.to_dict()
    wh = context.resize_max_wh(division=16)
    processed_image = pipe.__call__(
        width=wh[0],
        height=wh[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        num_inference_steps=context.num_inference_steps,
        generator=generator,
        guidance_scale=context.guidance_scale,
    ).images[0]

    processed_image = context.resize_image_to_max_wh(processed_image)
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
    image = context.load_image(division=16)
    mask = context.load_mask(division=16)
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


def diffusers_upscale_call(pipe, context: Context):

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


def optimize_pipeline(pipe):
    # Override the safety checker
    def dummy_safety_checker(images, **kwargs):
        return images, [False] * len(images)

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()  # Enable VAE tiling to improve memory efficiency
    pipe.enable_attention_slicing("auto")  # Enable attention slicing for faster inference
    pipe.safety_checker = dummy_safety_checker

    return pipe


quantization_config = BitsAndBytesConfig(load_in_8bit=True)


def get_t5_quantized(model_id):

    return T5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder_3",
        quantization_config=quantization_config,
    )

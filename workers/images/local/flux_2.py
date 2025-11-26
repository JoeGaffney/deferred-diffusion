import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from PIL import Image
from transformers import Mistral3ForConditionalGeneration

from common.config import IMAGE_CPU_OFFLOAD, IMAGE_TRANSFORMER_PRECISION
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    offload = True

    transformer = get_quantized_model(
        model_id="diffusers/FLUX.2-dev-bnb-4bit",
        subfolder="transformer",
        model_class=Flux2Transformer2DModel,
        target_precision=16,
        torch_dtype=torch.bfloat16,
        offload=offload,
    )
    text_encoder = get_quantized_model(
        model_id="black-forest-labs/FLUX.2-dev",
        subfolder="text_encoder",
        model_class=Mistral3ForConditionalGeneration,
        target_precision=8,
        torch_dtype=torch.bfloat16,
        offload=offload,
    )

    pipe = Flux2Pipeline.from_pretrained(
        model_id, text_encoder=text_encoder, transformer=transformer, torch_dtype=torch.bfloat16
    )

    # allways offload as heavy models
    return optimize_pipeline(pipe, offload=offload)


def text_to_image_call(context: ImageContext):
    model_id = "black-forest-labs/FLUX.2-dev"
    pipe = get_pipeline(model_id)
    prompt = context.data.cleaned_prompt

    # gather all possible reference images
    reference_images = []
    if context.mask_image and context.color_image:
        prompt = (
            "Use image 1 for the mask region for inpainting. And use image 2 for the base image only alter the mask region and aim for a seemless blend. "
            + prompt
        )
        reference_images.append(context.mask_image)
        reference_images.append(context.color_image)
    else:
        if context.color_image:
            reference_images.append(context.color_image)

        for current in context.get_reference_images():
            if current is not None:
                reference_images.append(current)

    processed_image = pipe.__call__(
        prompt=prompt,
        image=None if len(reference_images) == 0 else reference_images[:3],  # max 3 reference images
        num_inference_steps=20,
        guidance_scale=2.5,
        height=context.height,
        width=context.width,
        generator=context.generator,
    ).images[0]

    return processed_image


def main(context: ImageContext) -> Image.Image:
    return text_to_image_call(context)

import torch
from diffusers import (
    AutoPipelineForInpainting,
    FluxKontextPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image
from transformers import T5EncoderModel

from common.config import (
    IMAGE_CPU_OFFLOAD,
    IMAGE_TRANSFORMER_PRECISION,
    VIDEO_TRANSFORMER_PRECISION,
)
from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_t5_text_encoder,
    optimize_pipeline,
)
from common.text_encoders import get_pipeline_flux_text_encoder
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    args = {}

    args["transformer"] = get_quantized_model(
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


def image_to_image_call(context: ImageContext):
    pipe = get_pipeline("black-forest-labs/FLUX.1-Kontext-dev")

    args = {
        "width": context.width,
        "height": context.height,
        "image": context.color_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }
    args = apply_prompt_embeddings(args, context.data.prompt, "")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        raise ValueError("Flux-Kontext does not support text-to-image generation. Use Flux instead.")
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return image_to_image_call(context)

    raise ValueError(f"Unknown mode: {mode}")

import torch
from diffusers import (
    AutoPipelineForInpainting,
    FluxKontextPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
)
from PIL import Image
from transformers import T5EncoderModel

from common.logger import logger
from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_t5_text_encoder,
    optimize_pipeline,
)
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=model_id,
        subfolder="transformer",
        model_class=FluxTransformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder_2"] = get_quantized_t5_text_encoder(8)

    pipe = FluxKontextPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )

    return optimize_pipeline(pipe, offload=LOW_VRAM)


def image_to_image_call(context: ImageContext):
    pipe = get_pipeline(context.data.model_path)

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "image": context.color_image,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
        # "max_area": 1024**2,
    }

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

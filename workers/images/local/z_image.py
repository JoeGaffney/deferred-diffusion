import torch
from diffusers import DiffusionPipeline, ZImagePipeline, ZImageTransformer2DModel
from PIL import Image
from transformers import AutoModelForCausalLM

from common.logger import task_log
from common.memory import is_memory_exceeded
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
    task_log_callback,
)
from images.context import ImageContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    transformer = get_quantized_model(
        model_id="Tongyi-MAI/Z-Image-Turbo",
        subfolder="transformer",
        model_class=ZImageTransformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    text_encoder = get_quantized_model(
        model_id="Qwen/Qwen3-4B",
        subfolder="",
        model_class=AutoModelForCausalLM,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    pipe = ZImagePipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    return optimize_pipeline(pipe, offload=is_memory_exceeded(15))


def text_to_image_call(context: ImageContext):
    pipe = get_pipeline("Tongyi-MAI/Z-Image-Turbo")

    processed_image = pipe.__call__(
        prompt=context.data.cleaned_prompt,
        num_inference_steps=9,
        guidance_scale=0.0,
        height=context.height,
        width=context.width,
        generator=context.generator,
        callback_on_step_end=task_log_callback(9),  # type: ignore
    ).images[0]

    return processed_image


def main(context: ImageContext) -> Image.Image:
    context.ensure_divisible(16)
    return text_to_image_call(context)

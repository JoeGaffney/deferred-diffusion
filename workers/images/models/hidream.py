import torch
from diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
from PIL import Image
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, T5EncoderModel

from common.logger import logger
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    optimize_pipeline,
)
from images.context import ImageContext, PipelineConfig

T5_MODEL_PATH = "black-forest-labs/FLUX.1-schnell"
LLAMA_MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"


@decorator_global_pipeline_cache
def get_pipeline(config: PipelineConfig):
    args = {}

    args["transformer"] = get_quantized_model(
        model_id=config.model_id,
        subfolder="transformer",
        model_class=HiDreamImageTransformer2DModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    args["text_encoder_3"] = get_quantized_model(
        model_id=T5_MODEL_PATH,
        subfolder="text_encoder_2",
        model_class=T5EncoderModel,
        target_precision=8,
        torch_dtype=torch.bfloat16,
    )

    args["text_encoder_4"] = get_quantized_model(
        model_id=LLAMA_MODEL_PATH,
        subfolder="",
        model_class=LlamaForCausalLM,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )
    args["text_encoder_4"].output_hidden_states = True
    args["text_encoder_4"].output_attentions = True

    args["tokenizer_4"] = PreTrainedTokenizerFast.from_pretrained(LLAMA_MODEL_PATH)

    pipe = HiDreamImagePipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.bfloat16,
        **args,
    )

    return optimize_pipeline(pipe, sequential_cpu_offload=False)


def text_to_image_call(context: ImageContext):
    pipe = get_pipeline(context.get_pipeline_config())

    args = {
        "width": context.width,
        "height": context.height,
        "prompt": context.data.prompt,
        "negative_prompt": context.data.negative_prompt,
        "num_inference_steps": context.data.num_inference_steps,
        "generator": context.generator,
        "guidance_scale": context.data.guidance_scale,
    }

    logger.info(f"Text to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def image_to_image_call(context: ImageContext):
    pipe = get_pipeline(context.get_pipeline_config())

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

    pipe, args = setup_controlnets_and_ip_adapters(pipe, context, args)

    logger.info(f"Image to image call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def inpainting_call(context: ImageContext):
    pipe = get_pipeline(context.get_pipeline_config())

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
    }

    logger.info(f"Inpainting call {args}")
    processed_image = pipe.__call__(**args).images[0]
    context.cleanup()

    return processed_image


def main(context: ImageContext) -> Image.Image:
    mode = context.get_generation_mode()

    if mode == "text_to_image":
        return text_to_image_call(context)
    elif mode == "img_to_img":
        return image_to_image_call(context)
    elif mode == "img_to_img_inpainting":
        return inpainting_call(context)

    raise ValueError(f"Unknown mode: {mode}")

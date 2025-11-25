import copy
import traceback
from typing import Any, Dict

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common.logger import log_pretty, logger
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from texts.context import TextContext
from utils.utils import (
    load_image_from_base64,
    load_video_into_file,
    time_info_decorator,
)


@decorator_global_pipeline_cache
def get_pipeline(model_id) -> Qwen2_5_VLForConditionalGeneration:
    model = get_quantized_model(
        model_id=model_id,
        subfolder="",
        model_class=Qwen2_5_VLForConditionalGeneration,
        target_precision=4,
        torch_dtype=torch.bfloat16,
    )

    return model.to("cuda")


@time_info_decorator
def get_processor(model_id):
    # can affect performance could be reduced further
    # ref original
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    min_pixels = 64 * 28 * 28
    max_pixels = 198 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, device_map="cpu", use_fast=True
    )

    return processor


def main(context: TextContext) -> str:
    model = get_pipeline("Qwen/Qwen2.5-VL-3B-Instruct")
    processor = get_processor("Qwen/Qwen2.5-VL-3B-Instruct")

    system_message: Dict[str, Any] = {
        "role": "system",
        "content": [{"type": "text", "text": context.data.full_system_prompt}],
    }
    message: Dict[str, Any] = {
        "role": "user",
        "content": [{"type": "text", "text": context.data.prompt}],
    }

    for image in context.data.images:
        pil_image = load_image_from_base64(image)
        message["content"].append(
            {
                "type": "input_image",
                "image": pil_image,
            }
        )

    for video in context.data.videos:
        video_path = load_video_into_file(video, model=context.model)
        if video_path:
            message["content"].append(
                {
                    "type": "input_video",
                    "video": f"file://{video_path}",
                }
            )

    logger.info(f"Running qwen with {len(context.data.images)} images and {len(context.data.videos)} videos")

    # Preparation for inference
    text = processor.apply_chat_template([system_message, message], tokenize=False, add_generation_prompt=True)

    # only reprocess the images and videos in the last message
    try:
        image_inputs, video_inputs = process_vision_info([message])
    except Exception as e:
        error_message = f"Error during vision info processing: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        raise Exception(error_message)

    output = ""
    try:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)  # Move inputs to the same device as the model

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output = output_text[0]
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        output = f"Error during inference: {e}"
        raise Exception(output)
    finally:
        inputs = inputs.to("cpu")  # Move inputs back to CPU
        del inputs, generated_ids, generated_ids_trimmed, output_text, image_inputs, video_inputs

    return output

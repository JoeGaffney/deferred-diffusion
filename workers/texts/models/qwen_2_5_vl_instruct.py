import copy
import traceback

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common.logger import log_pretty, logger
from common.pipeline_helpers import decorator_global_pipeline_cache, get_quantized_model
from texts.context import TextContext
from utils.utils import load_image_from_base64


# @decorator_global_pipeline_cache
def get_pipeline(model_id):
    model = get_quantized_model(
        model_id=model_id,
        subfolder="",
        model_class=Qwen2_5_VLForConditionalGeneration,
        target_precision=4,
        torch_dtype=torch.float16,
    )
    model.to("cpu")  # Ensure model is on CPU initially

    logger.warning(f"Loaded pipeline {model_id}")
    return model


def get_proccesor(model_id):
    # can affect performance could be reduced further
    # ref original
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    min_pixels = 64 * 28 * 28
    max_pixels = 198 * 28 * 28

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, device_map="cpu", use_fast=True
    )

    logger.warning(f"Loaded processor pipeline {model_id}")
    return processor


def main(context: TextContext):
    model = get_pipeline(context.data.model_path)
    processor = get_proccesor(context.data.model_path)

    messages = [message.model_dump() for message in context.data.messages]
    original_messages = copy.deepcopy(messages)

    # apply image and video to last message
    last_message = messages[-1]
    for image in context.data.images:
        last_message_content = last_message.get("content", [])
        pil_image = load_image_from_base64(image)
        last_message_content.append(
            {
                "type": "image",
                "image": pil_image,
            }
        )
    logger.warning(f"Running qwen with {len(context.data.images)} images and {len(context.data.videos)} videos")

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # only reprocess the images and videos in the last message
    try:
        image_inputs, video_inputs = process_vision_info([last_message])
    except Exception as e:
        error_message = f"Error during vision info processing: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        raise Exception(error_message)

    output = ""
    try:
        model = model.to("cuda")
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
        model = model.to("cpu")  # Move model back to CPU
        del inputs, generated_ids, generated_ids_trimmed, output_text, image_inputs, video_inputs

    # we keep only the original as we may have altered adding video and image to the last message
    chain_of_thought = original_messages
    chain_of_thought.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": output,
                }
            ],
        }
    )
    result = {
        "response": output,
        "chain_of_thought": chain_of_thought,
    }

    log_pretty("qwen result", result)
    return result

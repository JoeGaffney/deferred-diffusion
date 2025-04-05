import copy
import traceback
from functools import lru_cache

from qwen_vl_utils import process_vision_info
from text.context import TextContext
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from utils.logger import logger
from utils.utils import free_gpu_memory


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    # can affect performance could be reduced further
    # ref original
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    min_pixels = 64 * 28 * 28
    max_pixels = 198 * 28 * 28

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, device_map="cpu", use_fast=True
    )
    logger.warning(f"Loaded pipeline {model_id}")
    return model, processor


def main(context: TextContext, flush_gpu_memory=True):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    model, processor = get_pipeline(context.data.model)
    messages = context.data.messages

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # only reprocess the images and videos in the last message
    last_message = messages[-1]
    try:
        image_inputs, video_inputs = process_vision_info([last_message])
    except Exception as e:
        error_message = f"Error during vision info processing: {e}\n{traceback.format_exc()}"
        logger.error(error_message)
        raise Exception(error_message)

    output = ""
    try:
        model = model.to("cuda")  # Move GPU
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

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
        if flush_gpu_memory:
            model = model.to("cpu")  # Move model back to CPU
            inputs = inputs.to("cpu")  # Move inputs back to CPU
            free_gpu_memory()

    chain_of_thought = copy.deepcopy(messages)
    chain_of_thought.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": output,
                }
            ],
        }
    )
    result = {
        "response": output,
        "chain_of_thought": chain_of_thought,
    }

    logger.info(result)
    return result

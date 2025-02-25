import copy
from functools import lru_cache
import os
import sys
import time
import traceback
from common.context import Context
from utils.pipeline_helpers import free_gpu_memory
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.logger import logger


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    # can affect performance could be reduced further
    # min_pixels = 256 * 28 * 28
    # max_pixels = 1280 * 28 * 28
    min_pixels = 64 * 28 * 28
    max_pixels = 198 * 28 * 28

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, device_map="cpu", use_fast=True
    )
    return model, processor


def main(context: Context, model_id="Qwen/Qwen2.5-VL-3B-Instruct", mode="text", flush_gpu_memory=True):
    model, processor = get_pipeline(model_id)
    messages = context.messages

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("messages: ")
    for message in messages:
        print(message)
    print("text: ", text)

    # only reprocess the images and videos in the last message
    last_message = messages[-1]
    try:
        image_inputs, video_inputs = process_vision_info([last_message])
    except Exception as e:
        error_message = f"Error during vision info processing: {e}\n{traceback.format_exc()}"
        context.log_error(error_message)
        return {"error": error_message}

    print(f"image_inputs: {image_inputs}")
    print(f"video_inputs: {video_inputs}")
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
        print(f"Error during inference: {e}")
        output = f"Error during inference: {e}"
    finally:
        if flush_gpu_memory:
            model = model.to("cpu")  # Move model back to CPU
            inputs = inputs.to("cpu")  # Move inputs back to CPU
            free_gpu_memory()

    chain_of_thought = copy.deepcopy(messages)
    chain_of_thought.append(
        {
            "role": "assistant",
            "content": {
                "type": "text",
                "text": output,
            },
        }
    )
    result = {
        "response": output,
        "chain_of_thought": chain_of_thought,
    }

    context.log(result)
    return result


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]
    input_image_path = f"file://../tmp/tornado_v001.JPG"
    pure_path = os.path.abspath("../tmp/tornado_v001.mp4")
    if os.path.exists(pure_path):
        print(f"exists {pure_path}")
    else:
        print(f"not exists {pure_path}")
        sys.exit(1)

    input_video_path = f"{pure_path}"
    input_image_path = f"../tmp/elf_v001.JPG"
    input_video_path = f"../tmp/tornado_v001.mp4"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image_path,
                },
                {
                    "type": "text",
                    "text": "Describe this image.",
                },
            ],
        }
    ]
    main(Context(messages=messages), flush_gpu_memory=True)

    # Wait for 30 seconds
    print("Sleeping for 2 seconds should flush gpu memory")
    time.sleep(2)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image_path,
                },
                {
                    "type": "text",
                    "text": "Give me a prompt for SD image generation to generate similar images.",
                },
            ],
        }
    ]
    main(Context(messages=messages), flush_gpu_memory=False)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": input_image_path,
                },
                {
                    "type": "video",
                    "video": input_video_path,
                },
                {
                    "type": "text",
                    "text": "Tell me the differences between the image and video. I want to know the differences not the content of each.",
                },
            ],
        }
    ]
    main(Context(messages=messages), flush_gpu_memory=False)

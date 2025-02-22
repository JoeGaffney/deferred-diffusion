from functools import lru_cache
import os
import time
from common.context import Context
from utils.pipeline_helpers import free_gpu_memory
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    # can affect performance could be reduced further
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=min_pixels, max_pixels=max_pixels, device_map="cpu", use_fast=True
    )
    return model, processor


def main(context: Context, model_id="Qwen/Qwen2.5-VL-3B-Instruct", mode="text"):
    model, processor = get_pipeline(model_id)

    input_image_path = f"file://{context.input_image_path}"
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
                    "text": context.prompt,
                },
            ],
        }
    ]

    print(messages)
    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    output = ""
    try:
        model = model.to("cuda")  # Move GPU
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        output = output_text[0]
    except Exception as e:
        print(f"Error during inference: {e}")
        output = f"Error during inference: {e}"
    finally:
        model = model.to("cpu")  # Move model back to CPU
        inputs = inputs.to("cpu")  # Move inputs back to CPU
        free_gpu_memory()

    return output


if __name__ == "__main__":
    output_name = os.path.splitext(os.path.basename(__file__))[0]

    main(
        Context(
            input_image_path="../tmp/tornado_v001.JPG",
            output_image_path=f"../tmp/output/{output_name}.png",
            prompt="Describe this image.",
            guidance_scale=7.5,
            num_inference_steps=10,
        )
    )

    # Wait for 30 seconds
    print("Sleeping for 10 seconds should flush gpu memory")
    time.sleep(10)

    main(
        Context(
            input_image_path="../tmp/tornado_v001.JPG",
            output_image_path=f"../tmp/output/{output_name}.png",
            prompt="Describe this image. And give me a prompt for SD image generation based on this.",
            guidance_scale=7.5,
            num_inference_steps=10,
        )
    )

from functools import lru_cache

import torch
from common.context import Context
from diffusers import LTXImageToVideoPipeline
from utils.logger import logger
from utils.utils import get_16_9_resolution


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = LTXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: Context):
    model_id = "Lightricks/LTX-Video"
    pipe = get_pipeline(model_id)
    image = context.load_image(division=32)
    generator = torch.Generator(device="cuda").manual_seed(context.seed)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.prompt,
        negative_prompt=context.negative_prompt,
        image=image,
        num_inference_steps=context.num_inference_steps,
        num_frames=context.num_frames,
        generator=generator,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


if __name__ == "__main__":
    width, height = get_16_9_resolution("540p")
    width, height = get_16_9_resolution("432p")

    main(
        Context(
            input_image_path="../tmp/tornado_v001.jpg",
            output_video_path="../tmp/output/tornado_v001_ltx_video.mp4",
            strength=0.2,
            prompt="Detailed, 8k, photorealistic, tornado, enchance keep original elements",
            num_inference_steps=50,
            max_width=width,
            max_height=height,
        )
    )

from functools import lru_cache

import torch
from diffusers import CogVideoXImageToVideoPipeline
from utils.logger import logger
from utils.utils import get_16_9_resolution
from video.context import VideoContext
from video.schemas import VideoRequest


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: VideoContext):
    context.model = "THUDM/CogVideoX1.5-5b-I2V"
    pipe = get_pipeline(context.model)
    image = context.load_image(division=16)
    generator = torch.Generator(device="cuda").manual_seed(42)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        image=image,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        generator=generator,
        num_videos_per_prompt=1,
    ).frames[0]

    processed_image_path = context.save_video(video, fps=16)
    return processed_image_path


if __name__ == "__main__":
    width, height = get_16_9_resolution("540p")

    main(
        VideoContext(
            VideoRequest(
                model="THUDM/CogVideoX1.5-5b-I2V",
                input_image_path="../test_data/color_v001.jpeg",
                output_video_path="../tmp/output/cog_video_x.mp4",
                strength=0.2,
                prompt="Tornado spinning in a farm land",
                negative_prompt="blurry, distorted",
                num_inference_steps=40,
                max_width=width,
                max_height=height,
                num_frames=81,
            )
        )
    )

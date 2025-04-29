from functools import lru_cache

import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
)

from common.logger import logger
from utils.utils import ensure_divisible, get_16_9_resolution
from videos.context import VideoContext
from videos.schemas import VideoRequest


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = LTXConditionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()
    pipe.enable_model_cpu_offload()

    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def image_to_video(context: VideoContext):
    context.model = "Lightricks/LTX-Video-0.9.5"
    pipe = get_pipeline(context.model)
    image = context.load_image(division=32)
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    condition1 = LTXVideoCondition(
        image=image,
        frame_index=0,
    )

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        conditions=[condition1],
        prompt=context.data.prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        generator=generator,
        guidance_scale=context.data.guidance_scale,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def text_to_video(context: VideoContext):
    context.model = "Lightricks/LTX-Video-0.9.5"
    pipe = get_pipeline(context.model)
    generator = torch.Generator(device="cuda").manual_seed(context.data.seed)
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    # Ensure dimensions are divisible by 32
    width = ensure_divisible(context.data.max_width, divisor=32)
    height = ensure_divisible(context.data.max_height, divisor=32)

    video = pipe.__call__(
        width=width,
        height=height,
        prompt=context.data.prompt,
        negative_prompt=negative_prompt,
        num_frames=context.data.num_frames,
        num_inference_steps=context.data.num_inference_steps,
        generator=generator,
        guidance_scale=context.data.guidance_scale,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def main(context: VideoContext):
    if context.data.input_image_path != "":
        return image_to_video(context)

    return text_to_video(context)


if __name__ == "__main__":
    width, height = get_16_9_resolution("540p")
    # width, height = get_16_9_resolution("432p")

    main(
        VideoContext(
            VideoRequest(
                model="Lightricks/LTX-Video",
                input_image_path="",
                output_video_path="../tmp/output/ltx_video_text_to_image.mp4",
                prompt="A young girl stands calmly in the foreground, looking directly at the camera, as a house fire rages in the background. Flames engulf the structure, with smoke billowing into the air. Firefighters in protective gear rush to the scene, a fire truck labeled '38' visible behind them. The girl's neutral expression contrasts sharply with the chaos of the fire, creating a poignant and emotionally charged scene.",
                num_inference_steps=50,
                guidance_scale=3,
                max_width=704,
                max_height=480,
                num_frames=96,
            )
        )
    )

    # main(
    #     VideoContext(
    #         VideoRequest(
    #             model="Lightricks/LTX-Video",
    #             input_image_path="../tmp/tornado_v001.jpg",
    #             output_video_path="../tmp/output/ltx_video.mp4",
    #             prompt="Tonrnado spinning in the sky, grass field moving in the wind, camera slowly zooming out",
    #             num_inference_steps=50,
    #             guidance_scale=3,
    #             max_width=width,
    #             max_height=height,
    #             num_frames=96,
    #         )
    #     )
    # )

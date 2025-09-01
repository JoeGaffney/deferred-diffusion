import torch
from diffusers.pipelines.ltx.pipeline_ltx_condition import (
    LTXConditionPipeline,
    LTXVideoCondition,
    LTXVideoTransformer3DModel,
)

from common.memory import LOW_VRAM
from common.pipeline_helpers import (
    decorator_global_pipeline_cache,
    get_quantized_model,
    get_quantized_t5_text_encoder,
)
from utils.utils import ensure_divisible, get_16_9_resolution, resize_image
from videos.context import VideoContext


@decorator_global_pipeline_cache
def get_pipeline(model_id):
    # NOTE don't actually see much difference in look with Q4 vs Q8
    transformer = get_quantized_model(
        model_id,
        subfolder="transformer",
        model_class=LTXVideoTransformer3DModel,
        target_precision=4 if LOW_VRAM else 4,
        torch_dtype=torch.bfloat16,
    )

    text_encoder = get_quantized_t5_text_encoder(8)

    pipe = LTXConditionPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )

    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()
    return pipe


def image_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    width, height = get_16_9_resolution("720p")
    image = context.image
    image = resize_image(image, 32, 1.0, width, height)

    condition1 = LTXVideoCondition(
        image=image,
        frame_index=0,
    )
    conditions = [condition1]

    if context.video_frames:
        video_condition = LTXVideoCondition(
            image=image,
            video=context.video_frames[: context.data.num_frames],
            frame_index=0,
        )
        conditions.append(video_condition)

    video = pipe.__call__(
        width=image.size[0],
        height=image.size[1],
        conditions=conditions,
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        generator=context.get_generator(),
        guidance_scale=1.0,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def text_to_video(context: VideoContext):
    pipe = get_pipeline("Lightricks/LTX-Video-0.9.7-distilled")

    width, height = get_16_9_resolution("720p")
    width = ensure_divisible(width, 32)
    height = ensure_divisible(height, 32)

    video = pipe.__call__(
        width=width,
        height=height,
        prompt=context.data.prompt,
        negative_prompt=context.data.negative_prompt,
        num_inference_steps=context.data.num_inference_steps,
        num_frames=context.data.num_frames,
        generator=context.get_generator(),
        guidance_scale=1.0,
    ).frames[0]

    processed_path = context.save_video(video)
    return processed_path


def main(context: VideoContext):
    if context.data.image:
        return image_to_video(context)

    return text_to_video(context)

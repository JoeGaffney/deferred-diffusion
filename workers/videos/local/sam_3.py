import random

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops
from sam3.model_builder import Sam3VideoPredictorMultiGPU, build_sam3_video_predictor

from common.logger import log_pretty, logger
from common.memory import free_gpu_memory
from common.pipeline_helpers import clear_global_pipeline_cache
from videos.context import VideoContext


def get_bpe_vocab_path() -> str:
    """seems missing in the sam3 package, so we add it here"""
    return hf_hub_download(
        repo_id="LanguageBind/LanguageBind", filename="open_clip/bpe_simple_vocab_16e6.txt.gz", repo_type="space"
    )


def shutdown(video_predictor: Sam3VideoPredictorMultiGPU, session_id):
    """Still seems that some memory exists after shut down, posssibly we need the transformers version"""
    video_predictor.handle_request(dict(type="close_session", session_id=session_id))
    video_predictor.shutdown()
    del video_predictor.model
    del video_predictor
    free_gpu_memory(message="Post SAM-3 Video Processing")


def main(context: VideoContext):
    """We can possibly move the implemenation to use transformers pipelines directly later."""
    if context.video_frames is None:
        raise ValueError("No video frames provided")

    # As its not in the pipeline cache, clear any existing encacse we need the space
    clear_global_pipeline_cache()
    video_predictor = build_sam3_video_predictor(bpe_path=get_bpe_vocab_path())

    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=context.video_frames,
        )
    )
    session_id = response["session_id"]

    # Add text prompt on frame 0
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=context.data.cleaned_prompt,
        )
    )
    output = response["outputs"]
    log_pretty(f"sam-3-video initial frame", output)

    # Use handle_stream_request to propagate masks across all frames
    all_frame_masks = {}

    for result in video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="both",
            start_frame_index=0,
            max_frame_num_to_track=None,
        )
    ):
        frame_idx = result["frame_index"]
        masks = result["outputs"].get("out_binary_masks")
        all_frame_masks[frame_idx] = masks

    # Convert dict to sorted list
    all_frame_masks_list = [all_frame_masks.get(i) for i in range(len(context.video_frames))]

    if not any(m is not None and len(m) > 0 for m in all_frame_masks_list):
        logger.warning(f'No masks detected for "{context.data.cleaned_prompt}"')
        shutdown(video_predictor, session_id)
        return context.save_video(context.video_frames)

    # Process each frame with colored masks
    processed_frames = []
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    original_frame_size = context.video_frames[0].size

    for frame_idx, masks in enumerate(all_frame_masks_list):
        # Get actual dimensions from the mask
        mask_width, mask_height = original_frame_size
        combined_mask = Image.new("RGB", (mask_width, mask_height), (0, 0, 0))

        if masks is None or len(masks) == 0:
            # Create blank frame when no mask detected
            processed_frames.append(combined_mask)
            continue

        for mask_idx, mask in enumerate(masks):
            color = base_colors[mask_idx] if mask_idx < len(base_colors) else (0, 0, 255)

            mask_uint8 = mask.astype("uint8") * 255
            mask_pil = Image.fromarray(mask_uint8, mode="L")

            color_layer = Image.new("RGB", (mask_width, mask_height), color)
            masked_color = Image.composite(
                color_layer,
                Image.new("RGB", (mask_width, mask_height), (0, 0, 0)),
                mask_pil,
            )

            combined_mask = ImageChops.add(combined_mask, masked_color, scale=1.0, offset=0)

        processed_frames.append(combined_mask)

    shutdown(video_predictor, session_id)
    processed_path = context.save_video(processed_frames, fps=24)
    return processed_path

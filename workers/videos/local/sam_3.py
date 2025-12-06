import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from transformers.models.sam3_video import Sam3VideoModel, Sam3VideoProcessor

from common.logger import log_pretty, logger
from common.memory import free_gpu_memory
from common.pipeline_helpers import clear_global_pipeline_cache
from videos.context import VideoContext


def main(context: VideoContext):
    """Using transformers implementation of SAM-3 for video segmentation. There is still some quaility isssues in the implementation currently."""
    if context.video_frames is None:
        raise ValueError("No video frames provided")

    # Clear existing pipeline cache as we need the space
    clear_global_pipeline_cache()

    # Initialize model and processor
    device = Accelerator().device  # "cuda" if torch.cuda.is_available() else "cpu"
    model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=context.video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
    )

    # Add text prompt to detect and track objects
    prompts = context.data.cleaned_prompt.split(",")
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompts,
    )

    processed_frames = []
    original_frame_size = context.video_frames[0].size

    def mask_color(idx):
        if idx == 0:
            return (255, 0, 0)
        elif idx == 1:
            return (0, 255, 0)
        else:
            return (0, 0, 255)

    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session, max_frame_num_to_track=None
    ):
        # postprocess to extract masks as numpy/boolean
        processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
        masks = processed_outputs["masks"]  # shape: (num_objects, H, W)

        h, w = masks.shape[1], masks.shape[2]
        combined = np.zeros((h, w, 3), dtype=np.uint8)

        for idx, mask in enumerate(masks):
            color = mask_color(idx)
            combined[mask.cpu().numpy()] = color

        # Resize to original frame if needed
        combined_img = Image.fromarray(combined).resize(original_frame_size, Image.Resampling.NEAREST)
        processed_frames.append(combined_img)

    # Clean up
    del model, processor, inference_session
    free_gpu_memory(message="Post SAM-3 Video Processing")

    processed_path = context.save_video(processed_frames, fps=24)
    return processed_path

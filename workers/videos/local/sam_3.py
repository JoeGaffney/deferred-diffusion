import random

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor

from common.logger import log_pretty, logger
from common.pipeline_helpers import clear_global_pipeline_cache
from videos.context import VideoContext


def overlay_masks(image, masks):
    image = image.convert("RGBA")

    # masks: torch.Tensor of shape (N, 1, H, W) or (N, H, W), bool
    if masks.ndim == 4:
        masks = masks[:, 0, :, :]  # (N, H, W)

    # to uint8 numpy [0, 255]
    masks = (masks.to(torch.uint8) * 255).cpu().numpy()  # (N, H, W)
    n_masks = masks.shape[0]

    rng = random.Random(42)  # fixed seed for reproducible colors
    colors = [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask_img = Image.fromarray(mask, mode="L")
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))  # 0.5 opacity
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)

    return image


# seems missing in the sam3 package, so we add it here
def get_bpe_vocab_path() -> str:
    return hf_hub_download(
        repo_id="LanguageBind/LanguageBind", filename="open_clip/bpe_simple_vocab_16e6.txt.gz", repo_type="space"
    )


def main(context: VideoContext):
    if context.video_frames is None:
        raise ValueError("No video frames provided")

    clear_global_pipeline_cache()
    bpe_path = get_bpe_vocab_path()
    logger.info(f"Using BPE vocab path: {bpe_path}")

    # Load the model
    video_predictor = build_sam3_video_predictor(bpe_path=bpe_path)
    text_prompt = context.data.cleaned_prompt

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
            text=text_prompt,
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
        logger.warning(f'No masks detected for "{text_prompt}"')
        video_predictor.handle_request(dict(type="close_session", session_id=session_id))
        return context.save_video(context.video_frames)

    # Process each frame with colored masks
    processed_frames = []
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for frame_idx, (original_frame, masks) in enumerate(zip(context.video_frames, all_frame_masks_list)):
        if masks is None or len(masks) == 0:
            processed_frames.append(original_frame)
            continue

        # Get actual dimensions from the mask
        mask_height, mask_width = masks[0].shape
        combined_mask = Image.new("RGB", (mask_width, mask_height), (0, 0, 0))

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

    # Close session - still need to free memory
    video_predictor.handle_request(dict(type="close_session", session_id=session_id))

    processed_path = context.save_video(processed_frames, fps=24)
    logger.info(f"Saved video to: {processed_path}")
    return processed_path

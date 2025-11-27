import random

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageChops
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from common.logger import log_pretty, logger
from common.pipeline_helpers import clear_global_pipeline_cache
from images.context import ImageContext


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


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No image provided")

    clear_global_pipeline_cache()
    bpe_path = get_bpe_vocab_path()
    logger.info(f"Using BPE vocab path: {bpe_path}")

    # Load the model
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.2)
    text_prompt = context.data.cleaned_prompt
    image_pil = context.color_image.convert("RGB")

    inference_state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
    # Get the masks, bounding boxes, and scores
    masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
    log_pretty(f"sam-3-image processed_dict", output)

    if not len(masks):
        logger.warning(f'No masks detected for "{text_prompt}", returning blank image.')
        return Image.new("RGB", (context.width, context.height), (0, 0, 0))

    # Overlay masks on the original image for visualization
    overlay_result = overlay_masks(image_pil, masks)
    context.save_image(overlay_result)

    image_masks = []
    for mask in masks:
        # mask: torch.Tensor of shape (1, H, W) or (H, W), bool
        if mask.ndim == 3:
            mask = mask.squeeze(0)  # (H, W)

        # convert bool tensor -> uint8 numpy array in [0, 255]
        mask_image = (mask.to(torch.uint8) * 255).cpu().numpy()
        processed_image = Image.fromarray(mask_image, mode="L")
        image_masks.append(processed_image)

    combined_clown_mask = Image.new("RGB", (context.width, context.height), (0, 0, 0))

    base_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
    ]
    fallback_color = (0, 0, 255)  # Blue for any remaining masks
    logger.info(f"Number of masks detected: {len(image_masks)}")

    # Apply colors to masks and combine them
    for i, mask in enumerate(image_masks):
        # Choose color: first 3 get base colors, the rest get fallback_color
        if i < len(base_colors):
            color = base_colors[i]
        else:
            color = fallback_color

        color_layer = Image.new("RGB", (context.width, context.height), color)

        masked_color = Image.composite(
            color_layer,
            Image.new("RGB", (context.width, context.height), (0, 0, 0)),
            mask,
        )

        combined_clown_mask = ImageChops.add(
            combined_clown_mask,
            masked_color,
            scale=1.0,
            offset=0,
        )

    return combined_clown_mask

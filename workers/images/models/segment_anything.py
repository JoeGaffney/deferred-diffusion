import numpy as np
from lang_sam import LangSAM
from PIL import Image, ImageChops

from common.logger import log_pretty, logger
from images.context import ImageContext


def main(context: ImageContext, mode="mask"):
    if context.color_image is None:
        raise ValueError("No image provided")

    model = LangSAM(sam_type="sam2.1_hiera_base_plus")
    image_pil = context.color_image.convert("RGB")
    text_prompt = context.data.prompt

    results = model.predict([image_pil], [text_prompt], box_threshold=0.3, text_threshold=0.25)
    model.to("cpu")  # Move model to CPU after inference

    results = results[0]
    log_pretty(f"processed_dict", results)

    if not len(results["masks"]):
        raise ValueError("No masks detected")

    image_masks = []
    for mask in results["masks"]:
        mask_image = (mask * 255).astype(np.uint8)  # Convert to 8-bit image
        processed_image = Image.fromarray(mask_image).convert("L")
        image_masks.append(processed_image)

    combined_clown_mask = Image.new("RGB", (context.width, context.height), (0, 0, 0))

    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
    ]
    if len(image_masks) > 3:
        colors = [
            (180, 0, 0),  # Soft Red
            (0, 180, 0),  # Soft Green
            (0, 0, 180),  # Soft Blue
            # (255, 0, 0),  # Red
            # (0, 255, 0),  # Green
            # (0, 0, 255),  # Blue
            (200, 100, 0),  # Orange
            (100, 0, 200),  # Violet
            (0, 200, 100),  # Aqua
            (200, 0, 100),  # Pink
            (100, 200, 0),  # Lime
            (0, 100, 200),  # Sky Blue
            (200, 180, 0),  # Gold (bright yellow but not pure)
        ]
    logger.info(f"Number of masks detected: {len(image_masks)}")
    logger.info(f"Colors used: {colors}")

    # Apply colors to masks and combine them
    for i, mask in enumerate(image_masks[:10]):
        # Create solid color image
        color_layer = Image.new("RGB", (context.width, context.height), colors[i])

        # Apply the mask to get only the part of the color layer we care about
        masked_color = Image.composite(color_layer, Image.new("RGB", (context.width, context.height), (0, 0, 0)), mask)

        # Add the masked color with clamping (scale=1.0 ensures no overflow beyond 255)
        combined_clown_mask = ImageChops.add(combined_clown_mask, masked_color, scale=1.0, offset=0)

    return combined_clown_mask

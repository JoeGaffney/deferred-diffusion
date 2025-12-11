import numpy as np
from lang_sam import LangSAM
from PIL import Image, ImageChops

from common.logger import log_pretty, task_log
from common.pipeline_helpers import clear_global_pipeline_cache
from images.context import ImageContext


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No image provided")

    clear_global_pipeline_cache()
    model = LangSAM(sam_type="sam2.1_hiera_base_plus")
    image_pil = context.color_image.convert("RGB")
    text_prompt = context.data.cleaned_prompt

    results = model.predict([image_pil], [text_prompt], box_threshold=0.3, text_threshold=0.25)
    model.sam.model.to("cpu")
    model.gdino.model.to("cpu")
    results = results[0]
    log_pretty(f"processed_dict", results)

    if not len(results["masks"]):
        raise ValueError("No masks detected")

    image_masks = []
    for mask in results["masks"]:
        mask_image = (mask * 255).astype(np.uint8)  # Convert to 8-bit image
        processed_image = Image.fromarray(mask_image).convert("L")
        image_masks.append(processed_image)

    task_log(f"Number of masks detected: {len(image_masks)}")

    combined_clown_mask = Image.new("RGB", (context.width, context.height), (0, 0, 0))
    base_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
    ]
    fallback_color = (0, 0, 255)  # Blue for any remaining masks

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

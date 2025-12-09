import random

import torch
from PIL import Image, ImageChops
from transformers.models.sam3 import Sam3Model, Sam3Processor

from common.logger import log_pretty, task_log
from common.memory import free_gpu_memory
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


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No image provided")

    clear_global_pipeline_cache()

    # Load the model and processor
    model = Sam3Model.from_pretrained("facebook/sam3").to("cuda")  # type: ignore
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    text_prompt = context.data.cleaned_prompt
    image_pil = context.color_image.convert("RGB")

    # Segment using text prompt
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process results
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()  # type: ignore
    )[0]
    log_pretty(f"processed_dict", results)

    masks = results["masks"]
    scores = results.get("scores", [])
    boxes = results.get("boxes", [])

    if not len(masks):
        del model, processor
        raise ValueError("No masks detected")

    # Convert masks to list of PIL images
    image_masks = []
    for mask in masks:
        # mask is already a numpy array or tensor from post_process_instance_segmentation
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask

        # Ensure mask is 2D
        if mask_np.ndim == 3:
            mask_np = mask_np.squeeze()

        # Convert to uint8 [0, 255]
        mask_uint8 = (mask_np * 255).astype("uint8")
        processed_image = Image.fromarray(mask_uint8, mode="L")
        image_masks.append(processed_image)

    task_log(f"Number of masks detected: {len(image_masks)}")

    # Overlay masks on the original image for visualization
    masks_tensor = torch.stack([m if isinstance(m, torch.Tensor) else torch.from_numpy(m) for m in masks])
    overlay_result = overlay_masks(image_pil, masks_tensor)
    context.save_image(overlay_result)

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

    del model, processor
    return combined_clown_mask

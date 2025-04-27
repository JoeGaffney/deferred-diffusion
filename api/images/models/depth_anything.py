from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, pipeline

from common.logger import logger
from images.context import ImageContext


# some other parts need switching to cpu, but this is the main one
def update_device(model, device):
    model.to(device)


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = AutoModelForDepthEstimation.from_pretrained(model_id)
    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: ImageContext, mode="depth"):
    context.model = "depth-anything/Depth-Anything-V2-Large-hf"
    if context.color_image is None:
        raise ValueError("No input image provided")

    image = context.color_image
    image_processor = AutoImageProcessor.from_pretrained(context.model)
    pipe = get_pipeline(context.model)
    update_device(pipe, "cuda")

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")
    # Move the inputs to the same device as the model (CUDA/CPU)
    inputs = {key: value.to("cuda").to(dtype=torch.bfloat16) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = pipe(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Convert tensor to PIL Image
    depth_map = prediction.squeeze().float().cpu().numpy()
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_map_8bit = (depth_map_normalized * 255).astype("uint8")
    processed_image = Image.fromarray(depth_map_8bit)

    processed_path = context.save_image(processed_image)

    update_device(pipe, "cpu")
    return processed_path

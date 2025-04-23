from functools import lru_cache

from transformers import pipeline

from common.logger import logger
from image.context import ImageContext


# some other parts need switching to cpu, but this is the main one
def update_device(pipe, device):
    pipe.model = pipe.model.to(device)
    # pipe.processor = pipe.model.to(device)
    # pipe.image_processor = pipe.model.to(device)


@lru_cache(maxsize=1)
def get_pipeline(model_id):
    pipe = pipeline(task="depth-estimation", model=model_id, device="cuda")
    update_device(pipe, "cpu")
    logger.warning(f"Loaded pipeline {model_id}")
    return pipe


def main(context: ImageContext, mode="depth"):
    context.model = "depth-anything/Depth-Anything-V2-Large-hf"
    if context.color_image is None:
        raise ValueError("No input image provided")

    pipe = get_pipeline(context.model)

    update_device(pipe, "cuda")
    processed_image = pipe(context.color_image)["depth"]
    update_device(pipe, "cpu")

    processed_path = context.save_image(processed_image)
    return processed_path

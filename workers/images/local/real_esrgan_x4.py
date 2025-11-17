import os

import torch
from RealESRGAN import RealESRGAN

from common.pipeline_helpers import clear_global_pipeline_cache
from images.context import ImageContext


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No input image provided")

    clear_global_pipeline_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealESRGAN(device, scale=4)

    # keep with our other models
    # MODEL_PATH = "weights/RealESRGAN_x4plus.pth"
    hf_home = os.getenv("HF_HOME", "")
    model_path = os.path.join(hf_home, "weights/RealESRGAN_x4plus.pth")
    model.load_weights(model_path, download=True)

    result = model.predict(context.color_image)

    # Move model to CPU to free GPU memory
    model.model.to("cpu")

    return result

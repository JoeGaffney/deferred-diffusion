import os

from RealESRGAN import RealESRGAN

from common.logger import task_log
from common.pipeline_helpers import clear_global_pipeline_cache
from images.context import ImageContext


def main(context: ImageContext):
    if context.color_image is None:
        raise ValueError("No input image provided")

    clear_global_pipeline_cache()
    model = RealESRGAN("cuda", scale=4)

    # keep with our other models
    hf_home = os.getenv("HF_HOME", "")
    model_path = os.path.join(hf_home, "weights/RealESRGAN_x4plus.pth")
    model.load_weights(model_path, download=True)

    result = model.predict(context.color_image)
    task_log("Image Super-Resolution completed")

    # cleanup
    model.model.to("cpu")
    del model.model
    del model
    return result

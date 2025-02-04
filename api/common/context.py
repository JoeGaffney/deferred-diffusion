from datetime import datetime
import os
from diffusers.utils import export_to_video
from api.utils.logger import logger


class Context:
    def __init__(
        self,
        image="",
        prompt="Detailed, 8k, photorealistic",
        negative_prompt="",
        strength=0.5,
        output_dir="./tmp/outputs",
        input_dir="./tmp",
        size_multiplier=0.5,
    ):
        self.image = image
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.size_multiplier = size_multiplier

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

    def save_image(self, image, name="processed", with_timestamp=True):
        os.path.join(self.output_dir, f"{name}.png")
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = os.path.join(self.output_dir, f"{name}_{timestamp}.png")

        image.save(path)

        self.log(f"Image saved at {path}")
        return path

    def save_video_with_timestamp(self, frames, name="processed", with_timestamp=True):
        os.path.join(self.output_dir, f"{name}.mp4")
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = os.path.join(self.output_dir, f"{name}_{timestamp}.mp4")

        export_to_video(frames, path, fps=7)

        self.log(f"Video saved at {path}")
        return path

    def get_input_image_path(self):
        if os.path.exists(os.path.join(self.input_dir, self.image)):
            return os.path.join(self.input_dir, self.image)

        raise FileNotFoundError(f"Image {self.image} not found in {self.input_dir}")

    def log(self, message):
        logger.info(message)

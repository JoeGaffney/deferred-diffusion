from datetime import datetime
import math
import os
from diffusers.utils import export_to_video, load_image
from utils.utils import assure_path_exists, save_copy_with_timestamp
from utils.logger import logger
from utils import device_info
import shutil


class Context:
    def __init__(
        self,
        image="",
        input_dir="../tmp",
        max_height=2048,
        max_width=2048,
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        num_frames=48,
        num_inference_steps=25,
        output_dir="../tmp/outputs",
        output_name="processed",
        prompt="Detailed, 8k, photorealistic",
        seed=42,
        strength=0.5,
    ):
        self.image = image
        self.input_dir = input_dir
        self.max_height = max_height
        self.max_width = max_width
        self.negative_prompt = negative_prompt
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.orig_height = 0
        self.orig_width = 0
        self.output_dir = output_dir
        self.output_name = output_name
        self.prompt = prompt
        self.seed = seed
        self.strength = strength

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.input_dir):
            os.makedirs(self.input_dir)

    def save_image(self, image, name_override=""):
        name = name_override if name_override != "" else self.output_name

        path = os.path.join(self.output_dir, f"{name}.png")
        image.save(path)
        self.log(f"Image saved at {path} size: {image.size}")

        save_copy_with_timestamp(path)
        return path

    def save_video(self, video, name_override="", fps=24):
        name = name_override if name_override != "" else self.output_name

        path = os.path.join(self.output_dir, f"{name}.mp4")
        export_to_video(video, path, fps=fps)
        self.log(f"Video saved at {path}")

        save_copy_with_timestamp(path)
        return path

    def get_input_image_path(self):
        if os.path.exists(os.path.join(self.input_dir, self.image)):
            return os.path.join(self.input_dir, self.image)

        raise FileNotFoundError(f"Image {self.image} not found in {self.input_dir}")

    def log(self, message):
        logger.info(message)

    def resize_image(self, image, division=32):
        # Ensure the new dimensions do not exceed max_width and max_height
        width = min(image.size[0], self.max_width)
        height = min(image.size[1], self.max_height)

        # Adjust width and height to be divisible by 32 or 8
        width = math.ceil(width / division) * division
        height = math.ceil(height / division) * division

        return image.resize((width, height))

    def resize_image_to_orig(self, image):
        return image.resize((self.orig_width, self.orig_height))

    def load_image(self, division=32):
        image = load_image(self.get_input_image_path())
        self.log(f"Image loaded from {self.get_input_image_path()} size: {image.size}")
        self.orig_width, self.orig_height = image.size

        tmp = self.resize_image(image, division)
        return tmp

from datetime import datetime
import math
import os
import subprocess
from diffusers.utils import export_to_video, load_image
from api.utils.logger import logger
import cv2
import numpy as np
from PIL import Image
import pyffmpeg
import tempfile


class Context:
    def __init__(
        self,
        image="",
        prompt="Detailed, 8k, photorealistic",
        # negative_prompt="jittery, distorted",
        negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
        strength=0.5,
        output_dir="./tmp/outputs",
        input_dir="./tmp",
        size_multiplier=0.5,
        seed=42,
        num_inference_steps=25,
        num_frames=48,
    ):
        self.image = image
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.size_multiplier = size_multiplier
        self.seed = seed
        self.num_inference_steps = num_inference_steps
        self.num_frames = num_frames

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

    def save_video(self, video, name="processed", with_timestamp=True, fps=24):
        os.path.join(self.output_dir, f"{name}.mp4")
        if with_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            path = os.path.join(self.output_dir, f"{name}_{timestamp}.mp4")

        export_to_video(video, path, fps=fps)

        self.log(f"Video saved at {path}")
        return path

    def get_input_image_path(self):
        if os.path.exists(os.path.join(self.input_dir, self.image)):
            return os.path.join(self.input_dir, self.image)

        raise FileNotFoundError(f"Image {self.image} not found in {self.input_dir}")

    def log(self, message):
        logger.info(message)

    def resize_image(self, image):
        width = image.size[0] * self.size_multiplier
        height = image.size[1] * self.size_multiplier

        # Adjust width and height to be divisible by 32
        width = math.ceil(width / 32) * 32
        height = math.ceil(height / 32) * 32

        return image.resize((width, height))

    def add_video_compression_alt(self, image):
        # Convert PIL image to NumPy array
        image_np = np.array(image)

        # Encode image using MJPEG codec
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Adjust quality as needed
        _, encoded_image = cv2.imencode(".jpg", image_np, encode_param)

        # Decode image back to NumPy array
        compressed_image_np = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

        # Convert NumPy array back to PIL image
        compressed_image = Image.fromarray(compressed_image_np)

        return compressed_image

    def add_video_compression(self, image):
        """Compresses a PIL image into a video and extracts the first frame from the video.
        Some of the video models need the compression noise of the seed image to generate moving images.
        """
        tmpdirname = tempfile.mkdtemp()

        input_path = os.path.join(tmpdirname, "temp_input.png")
        output_video_path = os.path.join(tmpdirname, "temp_output.mp4")
        output_image_path = os.path.join(tmpdirname, "frame_output.png")

        # Save the PIL image to a temporary file
        image.save(input_path)

        # Use pyffmpeg to compress the image into a video
        ff = pyffmpeg.FFmpeg()
        ff.options(f"-i {input_path} -vcodec libx264 -crf 23 {output_video_path}")

        # Extract the first frame from the video
        ff.options(f"-i {output_video_path} -vf select='eq(n\,0)' -q:v 3 {output_image_path}")

        # Read the extracted frame back into a PIL image
        compressed_image = Image.open(output_image_path)
        return compressed_image

    def load_image(self, add_compression=False):
        image = load_image(self.get_input_image_path())
        self.log(f"Image loaded from {self.get_input_image_path()} size: {image.size}")
        if add_compression:
            tmp = self.add_video_compression(image)
            self.save_image(tmp, "compressed")
            return self.resize_image(tmp)

        return self.resize_image(image)

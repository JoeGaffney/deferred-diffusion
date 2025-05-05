import tempfile
from typing import Literal

import requests
import torch
from diffusers.utils import export_to_video

from common.logger import logger
from utils.utils import load_image_if_exists
from videos.schemas import VideoRequest


class VideoContext:
    def __init__(self, data: VideoRequest):
        self.data = data
        self.model = data.model
        self.image = load_image_if_exists(data.image)

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        width, height = self.image.size
        if width > height:
            return "landscape"
        elif width < height:
            return "portrait"
        return "square"

    def save_video(self, video, fps=24):
        path = tempfile.NamedTemporaryFile(suffix=".mp4").name
        path = export_to_video(video, output_video_path=path, fps=fps)
        logger.info(f"Video saved at {path}")
        return path

    def save_video_url(self, url):
        path = tempfile.NamedTemporaryFile(suffix=".mp4").name

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path

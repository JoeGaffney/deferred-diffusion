import copy
import tempfile
from typing import Literal

import requests
import torch
from diffusers.utils import export_to_video

from common.logger import logger
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    load_image_if_exists,
    load_video_frames_if_exists,
)
from videos.schemas import VideoRequest


class VideoContext:
    def __init__(self, data: VideoRequest):
        self.data = data
        self.width = copy.copy(data.width)
        self.height = copy.copy(data.height)
        self.image = load_image_if_exists(data.image)
        if self.image:
            self.width, self.height = self.image.size

        self.video_frames = load_video_frames_if_exists(data.video)
        self.image_last_frame = load_image_if_exists(data.image_last_frame)

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def ensure_divisible(self, value: int):
        # Adjust width and height to be divisible by the specified value
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)
        if self.image:
            self.image = self.image.resize([self.width, self.height])
        if self.image_last_frame:
            self.image_last_frame = self.image_last_frame.resize([self.width, self.height])

    def long_video(self) -> bool:
        return self.data.num_frames > 100

    def tmp_video_path(self):
        return tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".mp4").name

    def save_video(self, video, fps=24):
        path = self.tmp_video_path()
        path = export_to_video(video, output_video_path=path, fps=fps, quality=9)
        logger.info(f"Video saved at {path}")
        return path

    def save_video_url(self, url):
        path = self.tmp_video_path()

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path

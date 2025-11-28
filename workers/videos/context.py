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
    mp4_to_base64_decoded,
)
from videos.schemas import VideoRequest

ONE_MB_IN_BYTES = 1 * 1024 * 1024


class VideoContext:
    def __init__(self, data: VideoRequest):
        self.model = data.model
        self.data = data
        self.width = copy.copy(data.width)
        self.height = copy.copy(data.height)
        self.image = load_image_if_exists(data.image)
        if self.image:
            self.width, self.height = self.image.size

        self.video_frames = load_video_frames_if_exists(data.video, model=self.model)
        self.last_image = load_image_if_exists(data.last_image)

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def is_720p_or_higher(self) -> bool:
        if self.width >= 1280 or self.height >= 1280:
            return True
        return False

    def is_1080p_or_higher(self) -> bool:
        if self.width >= 1920 or self.height >= 1920:
            return True
        return False

    def get_flow_shift(self) -> float:
        return 5.0 if self.is_720p_or_higher() else 3.0

    def ensure_divisible(self, value: int):
        # Adjust width and height to be divisible by the specified value
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)
        if self.image:
            self.image = self.image.resize([self.width, self.height])
        if self.last_image:
            self.last_image = self.last_image.resize([self.width, self.height])

    def ensure_frames_divisible(self, current_frames, divisor: int = 4) -> int:
        return ((current_frames - 1) // divisor) * divisor + 1

    def duration_in_seconds(self, fps=24) -> int:
        return max(1, int(self.data.num_frames / fps))

    def tmp_video_path(self, model="") -> str:
        return tempfile.NamedTemporaryFile(dir=get_tmp_dir(model), suffix=".mp4").name

    def save_video(self, video, fps=24):
        path = self.tmp_video_path(self.model)
        path = export_to_video(video, output_video_path=path, fps=fps, quality=9)
        logger.info(f"Video saved at {path}")
        return path

    def save_video_url(self, url):
        path = self.tmp_video_path(model=self.model)

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logger.info(f"Video saved at {path}")
        else:
            raise Exception(f"Failed to download file. Status code: {response.status_code}")

        return path

    def get_compressed_video(self, fps=24, mb_limit=15) -> str:
        if not self.data.video:
            raise ValueError("No video available.")

        if len(self.data.video) < mb_limit * ONE_MB_IN_BYTES:
            logger.info(
                f"Video size {len(self.data.video)/(ONE_MB_IN_BYTES):.2f} MB is within limit ({mb_limit} MB), no compression needed."
            )
            return self.data.video

        if not self.video_frames:
            raise ValueError("No video frames available.")

        tmp_path = self.tmp_video_path(model=self.model)
        path = export_to_video(self.video_frames, output_video_path=tmp_path, fps=fps, quality=9)
        logger.info(f"Compressed video saved at {path}")

        # convert and check size once more abort if still too large
        compressed_video = mp4_to_base64_decoded(path)
        if len(compressed_video) >= mb_limit * ONE_MB_IN_BYTES:
            raise ValueError(
                f"Compressed video size {len(compressed_video)/(ONE_MB_IN_BYTES):.2f} MB still exceeds limit of {mb_limit} MB."
            )

        return compressed_video

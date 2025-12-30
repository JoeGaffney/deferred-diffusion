import copy
import tempfile
from typing import Literal

import requests
import torch
from diffusers.utils import export_to_video

from common.logger import logger, task_log
from utils.utils import (
    ensure_divisible,
    get_tmp_dir,
    image_crop,
    image_resize,
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

        task_log(
            f"Context created {self.model}, {self.width}x{self.height}",
        )

    def get_generator(self, device="cuda"):
        return torch.Generator(device=device).manual_seed(self.data.seed)

    def get_mega_pixels(self) -> float:
        """Calculate the megapixels based on width and height.
        480p (854x480): ~0.41 MP
        580p (1024x576): ~0.59 MP
        720p (1280x720): ~0.92 MP
        1080p (1920x1080): ~2.07 MP
        1440p (2560x1440): ~3.69 MP
        4K (3840x2160): ~8.29 MP
        """
        return (self.width * self.height) / 1_000_000

    def rescale_to_max_megapixels(self, max_mp: float):
        """Rescale width and height to ensure the total megapixels do not exceed max_mp."""
        current_mp = self.get_mega_pixels()
        if current_mp <= max_mp:
            return  # Already under the limit

        scale_factor = (max_mp / current_mp) ** 0.5  # sqrt because area scales with both width and height
        self.width = int(self.width * scale_factor)
        self.height = int(self.height * scale_factor)

        if self.image:
            self.image = image_resize(self.image, (self.width, self.height))
        if self.last_image:
            self.last_image = image_resize(self.last_image, (self.width, self.height))
        if self.video_frames:
            for frame_idx in range(len(self.video_frames)):
                self.video_frames[frame_idx] = image_resize(self.video_frames[frame_idx], (self.width, self.height))

    def ensure_divisible(self, value: int):
        """Adjust width and height to be divisible by the specified value."""
        self.width = ensure_divisible(self.width, value)
        self.height = ensure_divisible(self.height, value)

        if self.image:
            self.image = image_crop(self.image, (self.width, self.height))
        if self.last_image:
            self.last_image = image_crop(self.last_image, (self.width, self.height))
        if self.video_frames:
            for frame_idx in range(len(self.video_frames)):
                self.video_frames[frame_idx] = image_crop(self.video_frames[frame_idx], (self.width, self.height))

    def get_dimension_type(self) -> Literal["square", "landscape", "portrait"]:
        """Determine the image dimension type based on width and height ratio."""
        if self.width > self.height:
            return "landscape"
        elif self.width < self.height:
            return "portrait"
        return "square"

    def ensure_frames_divisible(self, current_frames, divisor: int = 4) -> int:
        return ((current_frames - 1) // divisor) * divisor + 1

    def duration_in_seconds(self, fps=24) -> int:
        return max(1, int(self.data.num_frames / fps))

    def tmp_video_path(self) -> str:
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

        tmp_path = self.tmp_video_path()
        path = export_to_video(self.video_frames, output_video_path=tmp_path, fps=fps, quality=9)
        logger.info(f"Compressed video saved at {path}")

        # convert and check size once more abort if still too large
        compressed_video = mp4_to_base64_decoded(path)
        if len(compressed_video) >= mb_limit * ONE_MB_IN_BYTES:
            raise ValueError(
                f"Compressed video size {len(compressed_video)/(ONE_MB_IN_BYTES):.2f} MB still exceeds limit of {mb_limit} MB."
            )

        return compressed_video

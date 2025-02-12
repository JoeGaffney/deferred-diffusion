from datetime import datetime
import os
import shutil
from typing import Literal, Tuple
from .logger import logger

Resolutions = Literal["1080p", "900p", "720p", "576p", "540p", "480p", "432p", "360p"]
resolutions_16_9 = {
    "1080p": (1920, 1080),  # by 8
    "900p": (1600, 900),
    "720p": (1280, 720),  # by 8
    "576p": (1024, 576),  # by 8 and 32
    "540p": (960, 540),
    "480p": (854, 480),
    "432p": (768, 432),  # by 8
    "360p": (640, 360),
}


def get_16_9_resolution(resolution: Resolutions) -> Tuple[int, int]:
    return resolutions_16_9.get(resolution, (960, 540))


def assure_path_exists(path):
    my_dir = os.path.dirname(path)
    if not os.path.exists(my_dir):
        try:
            os.makedirs(my_dir)
        except Exception as e:
            print(e)
            pass


def save_copy_with_timestamp(path):
    if os.path.exists(path):
        directory, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        # Create the timestamped path
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        timestamp_path = os.path.join(directory, "tmp", f"{name}_{timestamp}{ext}")
        assure_path_exists(timestamp_path)

        shutil.copy(path, timestamp_path)

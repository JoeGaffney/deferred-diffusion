import base64
import tempfile
from io import BytesIO
from typing import List

from PIL import Image
from pydantic.types import Base64Bytes

from common.logger import logger, task_log
from utils.utils import get_tmp_dir
from workflows.schemas import WorkflowOutput, WorkflowRequest


class WorkflowContext:
    def __init__(self, data: WorkflowRequest):
        self.data = data

        task_log(
            f"WorkflowContext created for workflow with {len(self.data.workflow)} nodes and {len(self.data.patches)} patches",
        )

    def save_image(self, image: bytes) -> str:
        """Save image bytes to temp file. Raises ValueError if invalid."""
        # Validate it's a real, uncorrupted image
        try:
            img = Image.open(BytesIO(image))
            img.verify()
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image data: {e}")

        # Only write if validation passed
        with tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".png", delete=False) as tmp_file:
            tmp_file.write(image)
            path = tmp_file.name
            logger.info(f"Image validated and saved at {path}")

        return path

    def save_video(self, video: bytes) -> str:
        """Save video bytes to temp file. Raises ValueError if invalid."""
        # Check MP4 magic bytes
        if len(video) < 32 or b"ftyp" not in video[:32]:
            raise ValueError("Invalid video format: missing MP4 signature")

        # Only write if validation passed
        with tempfile.NamedTemporaryFile(dir=get_tmp_dir(), suffix=".mp4", delete=False) as tmp_file:
            tmp_file.write(video)
            path = tmp_file.name
            logger.info(f"Video validated and saved at {path}")

        return path

    def validate_and_save_output(self, output: WorkflowOutput) -> str:
        path = ""
        if output.data_type == "image" and output.base64_data:
            path = self.save_image(output.base64_data)
        elif output.data_type == "video" and output.base64_data:
            path = self.save_video(output.base64_data)

        return path

import tempfile

from common.logger import logger, task_log
from utils.utils import get_tmp_dir
from workflows.schemas import WorkflowRequest


class WorkflowContext:
    def __init__(self, data: WorkflowRequest):
        self.data = data

        task_log(
            f"WorkflowContext created for workflow with {len(self.data.workflow)} nodes and {len(self.data.patches)} patches",
        )

    def save_image(self, image):
        # Create a temporary file with .png extension
        with tempfile.NamedTemporaryFile(dir=get_tmp_dir("comfy-workflow"), suffix=".png", delete=False) as tmp_file:
            # tmp_file will be closed automatically when exiting the with block
            image.save(tmp_file, format="PNG")
            path = tmp_file.name
            logger.info(f"Image saved at {path}")

        return path

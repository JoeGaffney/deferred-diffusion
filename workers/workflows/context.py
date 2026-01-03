from pathlib import Path

from common.config import settings
from common.logger import get_task_id, logger, task_log
from workflows.schemas import WorkflowRequest


class WorkflowContext:
    def __init__(self, data: WorkflowRequest, task_id: str = get_task_id()):
        self.data = data
        self.task_id = task_id
        self.valid_extensions = [".png", ".mp4", ".exr"]

        task_log(
            f"WorkflowContext created for workflow with {len(self.data.workflow)} nodes and {len(self.data.patches)} patches",
        )

    def is_extension_valid(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.valid_extensions)

    def save_output(self, base64_bytes: bytes, filename: str) -> Path:
        # NOTE possibly we should validate the bytes here (e.g., try to open as image/video)
        if not self.is_extension_valid(filename):
            raise ValueError(f"Invalid file extension for output: {filename}")

        # deterministic relative path
        rel_path = Path("comfy-workflow") / f"{self.task_id}-{filename}"
        abs_path = settings.storage_dir / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(abs_path, "wb") as f:
                f.write(base64_bytes)
            logger.info(f"File saved at {abs_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save file at {abs_path}: {e}")

        return abs_path

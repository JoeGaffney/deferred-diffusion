import logging


class WorkerContext:
    """
    Provides context to execution tasks (e.g. logging, accessing mounted file systems,
    checking memory limits, etc).
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.logger = logging.getLogger(f"WorkerContext-{task_id}")

    def download_file(self, url: str, local_path: str):
        """Stub for pulling files before inference"""
        self.logger.info(f"Downloading {url} to {local_path}")

    def check_vram(self):
        """Stub to check available GPU VRAM"""
        return 24.0  # gigabytes mock
        return 24.0  # gigabytes mock
        return 24.0  # gigabytes mock
        return 24.0  # gigabytes mock
        return 24.0  # gigabytes mock

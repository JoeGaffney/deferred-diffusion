import logging
import pprint

from celery import current_task

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger object
logger = logging.getLogger(__name__)


def log_pretty(message, obj):
    """Utility function to pretty print objects in logs."""
    logger.info(message + "\n%s", pprint.pformat(obj, indent=1, width=120, sort_dicts=False))


def task_log(message: str):
    logger.info(message)
    task = current_task
    if not task or not hasattr(task, "update_state"):
        return

    # Use a per-task in-memory meta dict
    meta_existing = getattr(task, "_meta", {})
    if not isinstance(meta_existing, dict):
        meta_existing = {}

    logs = meta_existing.get("logs", [])
    if not isinstance(logs, list):
        logs = []

    logs.append(message)

    # Update meta and persist in-memory
    meta = {**meta_existing, "logs": logs}
    task._meta = meta  # type: ignore

    try:
        task.update_state(state="STARTED", meta=meta)  # type: ignore
    except Exception as e:
        logger.error(f"Failed to update task state: {e}")


def get_task_logs() -> list[str]:
    """Get accumulated logs for the current task."""
    task = current_task
    if not task:
        return []

    meta = getattr(task, "_meta", {})
    if isinstance(meta, dict):
        logs = meta.get("logs", [])
        if isinstance(logs, list):
            return logs
    return []

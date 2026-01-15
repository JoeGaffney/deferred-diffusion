import logging
import pprint
import uuid

from celery import current_task

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create a logger object
logger = logging.getLogger(__name__)


def log_pretty(message, obj):
    """Utility function to pretty print objects in logs."""
    logger.info(message + "\n%s", pprint.pformat(obj, indent=1, width=120, sort_dicts=False))


def task_log(message: str, log_to_logger: bool = True):
    if log_to_logger:
        logger.info(message)

    task = current_task
    if not task or not hasattr(task, "update_state"):
        return

    # Check if this is a new task by comparing task IDs
    current_id = getattr(task.request, "id", None)
    stored_id = getattr(task, "_logged_task_id", None)

    if stored_id != current_id:
        # New task - reset logs
        task._meta = {"logs": []}
        task._logged_task_id = current_id

    meta_existing = getattr(task, "_meta", {})
    logs = meta_existing.get("logs", [])
    logs.append(message)

    meta = {**meta_existing, "logs": logs}
    task._meta = meta

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


def get_task_id() -> str:
    """Get the current task ID. Falls back to a new UUID if not in a task context."""
    task_id: str = str(uuid.uuid4())

    task = current_task
    if not task:
        return task_id
    return getattr(task.request, "id", task_id)

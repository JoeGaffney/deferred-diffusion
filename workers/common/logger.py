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


# NOTE ref how to update task progress in Celery
def update_progress(progress, status=None, **extra_meta):
    task = current_task
    if not task or not hasattr(task, "update_state"):
        logger.error("No current task context available. Skipping update.")
        return

    meta = {"progress": progress}
    if status:
        meta["status"] = status
    meta.update(extra_meta)

    logger.info(f"Updating task progress: {progress}%, status: {status}")
    task.update_state(state="PROGRESS", meta=meta)

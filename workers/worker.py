import os
import time

import torch.multiprocessing as mp
from celery import Celery, Task

from common.logger import logger

# Set multiprocessing start method to 'spawn'
# This MUST be done at the top of the file, before any other code CUDA related code
# mp.set_start_method("spawn", force=True)


class BaseTask(Task):
    abstract = True  # Makes this a base class, not registered as a task

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failures globally"""
        # Log the error
        logger.error(f"Task {task_id} failed: {exc}")
        # You could add notification logic here (email, Slack, etc.)

        # Call parent handler
        super().on_failure(exc, task_id, args, kwargs, einfo)

    # def on_retry(self, exc, task_id, args, kwargs, einfo):
    #     """Handle task retries globally"""
    #     logger.info(f"Task {task_id} retrying: {exc}")
    #     super().on_retry(exc, task_id, args, kwargs, einfo)


celery_app = Celery(__name__, task_cls=BaseTask)
celery_app.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery_app.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


# Set default task settings for all tasks
celery_app.conf.task_acks_late = True  # Tasks acknowledged after execution
celery_app.conf.task_reject_on_worker_lost = True  # Requeue task if worker crashes
celery_app.conf.task_time_limit = 1800  # Time limit in seconds (30 minutes)
celery_app.conf.task_soft_time_limit = 1500  # Soft time limit (25 minutes)
celery_app.conf.task_send_sent_event = True  # Critical for seeing pending tasks
celery_app.conf.worker_prefetch_multiplier = 1  # Don't prefetch too many tasks

# Set up global retry behavior
celery_app.conf.task_default_retry_delay = 30  # Default retry delay (30 seconds)
celery_app.conf.task_max_retries = 1  # Default max retries

# Track all task states for better visibility
celery_app.conf.task_track_started = True  # Track when tasks are started
celery_app.conf.worker_send_task_events = True  # Send task-related events

# NOTE import task modules so they're registered with Celery
import images.tasks
import texts.tasks
import videos.tasks

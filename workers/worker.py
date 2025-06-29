import os
import time

from celery import Celery, Task

from common.logger import logger


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


celery_app = Celery(
    "deferred-diffusion",
    task_cls=BaseTask,
    broker=os.getenv("CELERY_BROKER_URL"),
    backend=os.getenv("CELERY_RESULT_BACKEND"),
)

celery_app.conf.task_track_started = True
celery_app.conf.worker_send_task_events = True
celery_app.conf.task_send_sent_event = True
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_acks_late = True  # Tasks acknowledged after execution
celery_app.conf.task_reject_on_worker_lost = True  # Requeue task if worker crashes

celery_app.conf.task_default_retry_delay = 30  # Default retry delay (30 seconds)
celery_app.conf.task_max_retries = 1  # Default max retries


# NOTE import task modules so they're registered with Celery
import images.tasks
import texts.tasks
import videos.tasks

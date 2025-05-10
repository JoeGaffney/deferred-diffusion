import os
import time

from celery import Celery

celery_app = Celery(__name__)
celery_app.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery_app.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

# Import task modules so they're registered with Celery
# import videos.tasks  # This will register all tasks in videos/tasks.py

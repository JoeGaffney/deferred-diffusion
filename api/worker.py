import os
import time

from celery import Celery

celery_app = Celery(
    "deferred-diffusion", broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND")
)

import os
import time

import torch.multiprocessing as mp
from celery import Celery

# Set multiprocessing start method to 'spawn'
# This MUST be done at the top of the file, before any other code CUDA related code
# mp.set_start_method("spawn", force=True)

celery_app = Celery(__name__)
celery_app.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery_app.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

# NOTE import task modules so they're registered with Celery
import images.tasks
import texts.tasks
import videos.tasks

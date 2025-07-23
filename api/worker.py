import os

from celery import Celery

celery_app = Celery(
    "deferred-diffusion", broker=os.getenv("CELERY_BROKER_URL"), backend=os.getenv("CELERY_RESULT_BACKEND")
)
celery_app.conf.broker_transport_options = {
    "socket_timeout": 15,  # Timeout for Redis socket ops (in seconds)
    "retry_on_timeout": True,  # Retry on timeout errors
    "max_retries": 2,  # Number of retries for send_task() delivery
}

celery_app.conf.update(
    result_backend_always_retry=False,  # Do not always retry result backend operations
    result_backend_max_retries=2,  # Number of retries for result backend operations
)

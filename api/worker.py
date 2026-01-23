from datetime import timedelta

from celery import Celery

from common.config import settings

celery_app = Celery("deferred-diffusion", broker=settings.celery_broker_url, backend=settings.celery_result_backend)
celery_app.conf.broker_transport_options = {
    "socket_timeout": 15,  # Timeout for Redis socket ops (in seconds)
    "retry_on_timeout": True,  # Retry on timeout errors
    "max_retries": 2,  # Number of retries for send_task() delivery
}

celery_app.conf.update(
    result_backend_always_retry=False,  # Do not always retry result backend operations
    result_backend_max_retries=2,  # Number of retries for result backend operations
)
celery_app.conf.result_expires = timedelta(days=settings.result_expires_days)

import importlib
import os

from celery import Celery

# Configure Celery
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery("deferred_diffusion_v3_workers", broker=BROKER_URL, backend=BACKEND_URL)

from external_worker.context import WorkerContext


def register_task(t_name: str):
    """Dynamically register task specifically for external_worker."""
    full_task_name = f"external_worker.{t_name}"

    @celery_app.task(name=full_task_name, bind=True)
    def wrapper(self, params_dict: dict):
        mod = importlib.import_module(f"external_worker.tasks.{t_name}.execute")

        execute_fn = getattr(mod, f"execute_{t_name}", getattr(mod, "execute", None))
        if not execute_fn:
            raise NotImplementedError(f"Task module for {t_name} is missing an execution function.")

        ctx = WorkerContext(task_id=self.request.id)
        return execute_fn(params_dict, ctx)


tasks_dir = os.path.join(os.path.dirname(__file__), "tasks")
for item in os.listdir(tasks_dir):
    item_path = os.path.join(tasks_dir, item)
    if os.path.isdir(item_path) and not item.startswith("__"):
        register_task(item)
    if os.path.isdir(item_path) and not item.startswith("__"):
        register_task(item)
    if os.path.isdir(item_path) and not item.startswith("__"):
        register_task(item)
    if os.path.isdir(item_path) and not item.startswith("__"):
        register_task(item)
    if os.path.isdir(item_path) and not item.startswith("__"):
        register_task(item)

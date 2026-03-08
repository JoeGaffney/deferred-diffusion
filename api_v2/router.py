import importlib
import os
import uuid

from celery import Celery
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared.common_schemas import TaskResponse, TaskStatus

router = APIRouter(tags=["Tasks"])

# Configure Celery to be able to send tasks to the worker
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")

celery_app = Celery("deferred_diffusion_v3_workers", broker=BROKER_URL, backend=BACKEND_URL)

# In a real setup, we'd query Redis for these
_mock_task_db = {}


def register_create_endpoint(worker_type: str, task_name: str, schema_class: type[BaseModel]):
    """
    Dynamically creates a POST endpoint for a specific task using its schema.
    """

    @router.post(
        f"/tasks/{worker_type}/{task_name}", response_model=dict, status_code=202, summary=f"Create {task_name} Task"
    )
    def create_task(request: schema_class):
        task_id = str(uuid.uuid4())

        # Store initial state
        _mock_task_db[task_id] = TaskResponse(task_id=task_id, status=TaskStatus.PENDING)

        # Send the task to Celery on the specific worker type's queue
        celery_app.send_task(
            name=f"{worker_type}.{task_name}", args=[request.model_dump()], task_id=task_id, queue=worker_type
        )

        return {"task_id": task_id, "status": "accepted"}


# Scan the worker directories to dynamically build the routes
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
worker_types = ["local_worker", "external_worker"]

for w_type in worker_types:
    tasks_dir = os.path.join(base_dir, w_type, "tasks")
    if os.path.exists(tasks_dir):
        for item in os.listdir(tasks_dir):
            item_path = os.path.join(tasks_dir, item)
            if os.path.isdir(item_path) and not item.startswith("__"):
                try:
                    mod = importlib.import_module(f"{w_type}.tasks.{item}.schemas")
                    # Find the parameter schema class
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseModel) and attr_name.endswith("Params"):
                            register_create_endpoint(w_type, item, attr)
                            break
                except ImportError:
                    pass


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str):
    """
    Gets the state of a previously submitted task.
    """
    # In a prod setup, you would check celery_app.AsyncResult(task_id) state
    task = _mock_task_db.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return task

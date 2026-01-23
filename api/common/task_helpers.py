from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import UUID

import httpx
from cachetools import TTLCache, cached
from celery.result import AsyncResult
from fastapi import HTTPException

from common.config import settings
from common.logger import logger
from common.redis_manager import redis_manager
from common.schemas import DeleteResponse, Identity, TaskStatus
from worker import celery_app


@cached(cache=TTLCache(maxsize=128, ttl=5))
def _get_task_info(task_id: str) -> Dict[str, Any]:
    """
    Fetch task information from Flower API.
    So we can use this to provide more detailed task status in the API responses.
    Instead of using celerys extended results feature which would use a lot of storage.
    """
    result = {}
    try:
        url = f"{settings.flower_url}/api/task/info/{task_id}"
        with httpx.Client() as client:
            response = client.get(url, timeout=5.0)
            response.raise_for_status()
            if response.status_code == 200:
                result = response.json()
    except Exception as e:
        logger.warning(f"Error fetching task info from Flower: {e}")
        return {}

    # Remove any empty or null fields
    result = {k: v for k, v in result.items() if v not in [None, "", [], {}]}

    # Always remove these fields
    exclude_keys = ["root", "root_id", "parent", "parent_id", "children", "result", "uuid", "clock"]
    for key in exclude_keys:
        result.pop(key, None)

    # Convert timestamps to ISO datetime strings
    timestamp_keys = ["received", "sent", "started", "rejected", "succeeded", "timestamp"]
    for key in timestamp_keys:
        if key in result and isinstance(result[key], (int, float)):
            try:
                result[key] = datetime.fromtimestamp(result[key], tz=timezone.utc).isoformat(timespec="seconds")
            except (ValueError, OverflowError, TypeError):
                # Use original value if conversion fails
                pass

    # NOTE possibly we could put into a fixed schema here instead of returning raw dict
    return result


def create_task(task_name: str, task_queue: str, payload: dict, identity: Identity) -> AsyncResult:
    """
    Unified helper to create a task in Celery.
    """
    try:
        return celery_app.send_task(
            task_name,
            queue=task_queue,
            args=[payload],
            kwargs=identity.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")


def get_task_detailed(id: UUID) -> tuple[AsyncResult, dict, list[str]]:
    """
    Fetches the task across current Redis storage (Broker and Result Backend).
    Returns (AsyncResult, task_info, initial_logs).
    Raises 404 if the task is not in Redis (either never existed or has expired).
    """

    def get_queue_position(task_id: str) -> Optional[str]:
        """
        Inner helper to check the broker and format the queue position log.
        """
        pos_data = redis_manager.get_queue_position(task_id)
        if pos_data:
            return f"Queue {pos_data.queue} position: {pos_data.position} / {pos_data.total}"

        return None

    result = AsyncResult(str(id), app=celery_app)
    logs = []

    # Celery reports waiting tasks as PENDING and also unknown tasks as PENDING.
    if result.status == TaskStatus.PENDING:
        queue_position = get_queue_position(str(id))
        if queue_position is None:
            # Truly not found
            raise HTTPException(status_code=404, detail="Task not found or has expired")

        # Keep the queue position logs to return to the user
        logs = [queue_position]
    else:
        # get the running logs of the task if available
        if result.info:
            if isinstance(result.info, dict):
                logs = result.info.get("logs", [])

    # Enrich with Flower metadata if available (metrics, worker info, etc)
    task_info = _get_task_info(str(id))

    return result, task_info, logs


def cancel_task(id: UUID) -> DeleteResponse:
    result = AsyncResult(str(id), app=celery_app)

    if result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
        return DeleteResponse(id=id, status=result.status, message="Task already completed")

    try:
        celery_app.control.revoke(str(id), terminate=True)
        # result.forget()  # Optional: removes result from backend/Flower after revoke
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

    return DeleteResponse(id=id, status=TaskStatus.REVOKED, message="Task cancellation requested")

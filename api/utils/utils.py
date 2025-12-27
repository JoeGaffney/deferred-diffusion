import asyncio
import time
from typing import Any
from uuid import UUID

from celery.result import AsyncResult
from fastapi import HTTPException

from common.schemas import DeleteResponse, TaskStatus


def cancel_task(id: UUID, celery_app) -> DeleteResponse:
    result = AsyncResult(str(id), app=celery_app)

    if result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
        return DeleteResponse(id=id, status=result.status, message="Task already completed")

    try:
        celery_app.control.revoke(str(id), terminate=True)
        # result.forget()  # Optional: removes result from backend/Flower after revoke
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

    return DeleteResponse(id=id, status=TaskStatus.REVOKED, message="Task cancellation requested")


def truncate_strings(data: Any, max_length: int = 100) -> Any:
    if isinstance(data, dict):
        return {k: truncate_strings(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_strings(item, max_length) for item in data]
    elif isinstance(data, str):
        return data if len(data) <= max_length else data[:max_length] + "..."
    else:
        return data

import asyncio
import time
from uuid import UUID

from celery.result import AsyncResult
from fastapi import HTTPException

from common.logger import logger
from common.schemas import DeleteResponse
from worker import celery_app


async def poll_until_resolved(id: str, timeout=300, poll_interval=3) -> AsyncResult:
    """
    Poll a Celery task until it's complete, max attempts reached, or timeout.

    Returns:
        AsyncResult object with task result/status
    """
    # Ensure id is a string
    if not isinstance(id, str):
        logger.warning(f"id is not a string, converting from {type(id)}")
        id = str(id)

    start_time = time.time()
    while time.time() - start_time < timeout:
        logger.info(f"Polling task {id}... {timeout - (time.time() - start_time)} seconds left")
        # Get task result
        result = AsyncResult(id, app=celery_app)

        if result.ready():
            if result.failed():
                error_msg = str(result.result)
                logger.error(f"Task {id} failed: {error_msg}")
                return result
            elif result.successful():
                return result

        # Wait before polling again
        await asyncio.sleep(poll_interval)

    logger.warning(f"Task {id} polling timeout ({timeout} seconds) reached")

    # Get final status before returning
    result = AsyncResult(id, app=celery_app)
    return result


def cancel_task(id: UUID, celery_app) -> DeleteResponse:
    result = AsyncResult(str(id), app=celery_app)

    if result.status in ["SUCCESS", "FAILURE", "REVOKED"]:
        return DeleteResponse(id=id, status=result.status, message="Task already completed")

    try:
        celery_app.control.revoke(str(id), terminate=True)
        # result.forget()  # Optional: removes result from backend/Flower after revoke
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling task: {str(e)}")

    return DeleteResponse(id=id, status="REVOKED", message="Task cancellation requested")

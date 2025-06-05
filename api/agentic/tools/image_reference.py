from fastapi import HTTPException

from texts.schemas import MessageContent, MessageItem, TextRequest, TextWorkerResponse
from utils.utils import poll_until_complete
from worker import celery_app


async def main(prompt: str, image_reference_image: str) -> str:
    print(f"Processing image reference with prompt: {prompt}")
    request = TextRequest(
        model="gpt-4o-mini",  # or "gpt-4.1-mini" for the other model
        messages=[
            MessageItem(
                role="user",
                content=[
                    MessageContent(type="input_text", text=prompt),
                ],
            ),
        ],
        images=[image_reference_image],
    )
    id = None
    try:
        create_result = celery_app.send_task("process_text_external", args=[request.model_dump()])
        id = create_result.id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating task: {str(e)}")

    if id is None:
        raise HTTPException(status_code=500, detail="Task ID is None, task creation failed.")

    result = await poll_until_complete(str(id))
    if result.successful():
        try:
            result_data = TextWorkerResponse.model_validate(result.result)
            print(f"Result: {result_data}")
            return result_data.response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error parsing result: {str(e)}")
    else:
        raise HTTPException(status_code=500, detail=f"Task failed with error: {str(result.result)}")

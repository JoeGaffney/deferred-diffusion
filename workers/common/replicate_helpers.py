import time
from typing import Any

import replicate
from diffusers.utils import load_image
from PIL import Image
from replicate.helpers import FileOutput

from common.logger import logger, task_log


def replicate_run(model_path: str, payload: dict[str, Any], poll_interval: int = 3) -> Any:
    task_log(
        f"Calling Replicate API {model_path}",
    )
    try:
        # Create prediction
        prediction = replicate.predictions.create(
            model=model_path,
            input=payload,
        )

        # Poll for status and stream logs
        last_log_position = 0
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            time.sleep(poll_interval)  # Small delay before next poll
            logger.info(f"Polling replicate for completion: {prediction.status}")
            prediction.reload()

            # Stream new logs if available
            if prediction.logs:
                new_logs = prediction.logs[last_log_position:]
                if new_logs:
                    for line in new_logs.splitlines():
                        if line.strip():
                            task_log(line.strip())
                    last_log_position = len(prediction.logs)

        # Check final status
        if prediction.status == "failed":
            raise RuntimeError(f"Replicate prediction failed: {prediction.error}")

        # NOTE: does not always return a URL, sometimes a FileOutput object
        output = prediction.output
    except Exception as e:
        raise RuntimeError(f"Error calling Replicate API {model_path}: {e}")

    task_log(
        f"Completed replicate call {model_path}",
    )
    return output


def process_replicate_image_output(output: Any) -> Image.Image:
    url = output
    if isinstance(output, FileOutput):
        url = output.url

    if not isinstance(url, str):
        raise ValueError(f"Incorrect output from replicate: {type(output)} {str(output)}")

    try:
        processed_image = load_image(url)
    except Exception as e:
        raise ValueError(f"Failed to process image from replicate: {str(e)} {output}")

    return processed_image


def process_replicate_video_output(output: Any) -> str:
    """Process replicate video output and return the URL for download."""
    url = output
    if isinstance(output, FileOutput):
        url = output.url

    if not isinstance(url, str):
        raise ValueError(f"Incorrect output from replicate: {type(output)} {str(output)}")

    return url

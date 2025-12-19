import base64
import os
import time
from typing import Any, Optional

import httpx

from common.logger import logger

# "http://comfy:8188"
COMFY_API_URL = os.getenv("COMFY_API_URL")

# Module-level client for connection pooling
_client: Optional[httpx.Client] = None


def get_client() -> httpx.Client:
    if COMFY_API_URL is None:
        raise RuntimeError("COMFY_API_URL environment variable is not set")

    """Get or create the module-level httpx client for ComfyUI API calls."""
    global _client
    if _client is None:
        _client = httpx.Client(
            base_url=COMFY_API_URL,
            timeout=httpx.Timeout(60.0),
            follow_redirects=True,
        )
    return _client


def close_client() -> None:
    """Close the module-level client if it exists."""
    global _client
    if _client is not None:
        _client.close()
        _client = None


def is_comfy_running(timeout: float = 5.0, attempts: int = 10) -> bool:
    """Check if ComfyUI is up by making an HTTP request with a timeout."""
    for attempt in range(attempts):
        try:
            client = get_client()
            response = client.get("/", timeout=timeout)
            return response.status_code == 200
        except Exception:
            logger.warning(f"ComfyUI {COMFY_API_URL} not responding, attempt {attempt + 1} of {attempts}")
            if attempt < attempts - 1:
                time.sleep(timeout)
            continue
    return False


def api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    """Send a workflow to ComfyUI for processing."""
    client = get_client()
    payload = {"prompt": workflow}
    response = client.post("/prompt", json=payload)
    response.raise_for_status()
    return response.json()


def api_history(prompt_id: str) -> dict[str, Any]:
    """Get the execution history for a specific prompt ID."""
    client = get_client()
    response = client.get(f"/history/{prompt_id}")
    response.raise_for_status()
    return response.json()


def api_image_upload(base64_str: str, subfolder: str, filename: str) -> str:
    """
    Upload an image to ComfyUI's input directory.

    Args:
        base64_str: Base64-encoded image data
        subfolder: Subdirectory within ComfyUI's input folder
        filename: Filename to save as

    Returns:
        Path in the format "subfolder/filename"
    """

    file_bytes = base64.b64decode(base64_str)

    files = {"image": (filename, file_bytes, "application/octet-stream")}
    data = {
        "subfolder": subfolder or "",
        "type": "input",
        "overwrite": "1",
    }

    client = get_client()
    response = client.post("/upload/image", files=files, data=data)
    response.raise_for_status()

    result = response.json()
    result_subfolder = result.get("subfolder")
    result_filename = result.get("name")

    if not result_subfolder or not result_filename:
        raise RuntimeError(f"ComfyUI upload response missing subfolder or filename: {result}")

    return f"{result_subfolder}/{result_filename}"


def api_free(unload_models: bool = True, free_memory: bool = False) -> None:
    """
    Trigger ComfyUI to release VRAM and/or unload models.

    Args:
        unload_models: Free VRAM only
        free_memory: Free CPU memory as well
    """
    if not is_comfy_running():
        logger.warning("ComfyUI is not running — skipping resource cleanup.")
        return

    # Set the free flags
    payload = {"unload_models": unload_models, "free_memory": free_memory}

    try:
        client = get_client()
        response = client.post("/free", json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to set resource cleanup flags: {e}")
        return

    # Run a dummy prompt to apply the cleanup
    dummy_workflow = {
        "2": {
            "inputs": {"filename_prefix": "ComfyUIDummy", "images": ["4", 0]},
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        "4": {
            "inputs": {"width": 32, "height": 32, "batch_size": 1, "color": 0},
            "class_type": "EmptyImage",
            "_meta": {"title": "EmptyImage"},
        },
    }

    try:
        prompt_response = api_prompt(dummy_workflow)
        prompt_id = prompt_response.get("prompt_id")
        if not prompt_id:
            logger.warning("ComfyUI cleanup job failed: no prompt_id returned")
            return
    except Exception as e:
        logger.warning(f"ComfyUI cleanup job failed: {e}")


def api_view(filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
    """
    Gets arbitary output data from comfy usually a image or video.

    Args:
        filename: Name of the image file
        subfolder: Subdirectory within the folder_type
        folder_type: Type of folder (usually "output")

    Returns:
        contents of the file as bytes
    """
    client = get_client()
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type,
        "channel": "raw",
    }
    response = client.get("/view", params=params)
    response.raise_for_status()

    try:
        encoded = base64.b64encode(response.content)
    except Exception as e:
        raise RuntimeError(f"Failed to encode ComfyUI view response: {e}") from e
    return encoded

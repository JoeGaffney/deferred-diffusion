import base64
import copy
import json
import os
import socket
import time
import urllib.request
import uuid
from io import BytesIO

from PIL import Image

from common.logger import log_pretty, logger
from workflows.context import WorkflowContext
from workflows.schemas import WorkflowRequest

COMFY_API_URL = os.getenv("COMFY_API_URL", f"http://127.0.0.1:8188")
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


def is_comfy_running(timeout=5, attempts=10) -> bool:
    """Check if ComfyUI is up by making an HTTP request with a timeout."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{COMFY_API_URL}", timeout=timeout) as response:
                return response.status == 200
        except Exception:
            logger.warning(f"ComfyUI {COMFY_API_URL} not responding, attempt {attempt + 1} of {attempts}")
            if attempt < attempts - 1:
                time.sleep(timeout)
            continue
    return False


def api_prompt(workflow):
    """Send a workflow to ComfyUI for processing."""
    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_API_URL}/prompt", data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read().decode("utf-8"))


def api_history(prompt_id):
    """Get the execution history for a specific prompt ID."""
    response = urllib.request.urlopen(f"{COMFY_API_URL}/history/{prompt_id}")
    return json.loads(response.read().decode("utf-8"))


def api_image_upload(base64_str, subfolder, filename) -> str:
    file_bytes = base64.b64decode(base64_str)

    boundary = uuid.uuid4().hex
    crlf = "\r\n"

    fields = {"subfolder": subfolder or "", "type": "input", "overwrite": "1"}  # e.g. "input"

    body = bytearray()

    for k, v in fields.items():
        body.extend(f"--{boundary}{crlf}".encode())
        body.extend(f'Content-Disposition: form-data; name="{k}"{crlf}{crlf}'.encode())
        body.extend(f"{v}{crlf}".encode())

    body.extend(f"--{boundary}{crlf}".encode())
    body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"{crlf}'.encode())
    body.extend(f"Content-Type: application/octet-stream{crlf}{crlf}".encode())
    body.extend(file_bytes)
    body.extend(crlf.encode())
    body.extend(f"--{boundary}--{crlf}".encode())

    req = urllib.request.Request(
        f"{COMFY_API_URL}/upload/image",
        data=bytes(body),
        method="POST",
    )
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

    response = urllib.request.urlopen(req)
    if response.status != 200:
        raise RuntimeError(f"Failed to upload image to ComfyUI: {response.status} {response.reason}")

    result = json.loads(response.read().decode("utf-8"))
    result_subfolder = result.get("subfolder")
    result_filename = result.get("name")

    if not result_subfolder or not result_filename:
        raise RuntimeError("ComfyUI upload response missing subfolder or filename", result)

    return f"{result_subfolder}/{result_filename}"  # return the filename path


def api_free(unload_models=True, free_memory=False):
    """Trigger ComfyUI to release VRAM and/or unload models. unload_models is VRAM only, free_memory is for CPU memory aswell."""
    if not is_comfy_running():
        logger.warning("ComfyUI is not running — skipping resource cleanup.")
        return

    # Set the free flags
    payload = {"unload_models": unload_models, "free_memory": free_memory}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_API_URL}/free", data=data)
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        logger.warning(f"Failed to set resource cleanup flags: {e}")
        return

    # Run a dummy prompt to apply the cleanup
    try:
        response = api_prompt(dummy_workflow)
        prompt_id = response.get("prompt_id")
        if not prompt_id:
            logger.warning("ComfyUI cleanup job failed: no prompt_id returned")
            return
    except Exception as e:
        logger.warning(f"ComfyUI cleanup job failed: {e}")


def api_view(filename, subfolder="", folder_type="output") -> Image.Image:
    """Get an image from ComfyUI's output directory."""
    url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}&channel=raw"
    response = urllib.request.urlopen(url)
    img_data = response.read()
    return Image.open(BytesIO(img_data))


def poll_until_resolved(prompt_id, timeout=300, poll_interval=1):
    """Wait for the ComfyUI prompt to complete processing."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = api_history(prompt_id)
        log_pretty(f"History {prompt_id}", history)

        if prompt_id in history and "status" in history[prompt_id]:
            status = history[prompt_id]["status"]
            if status.get("status_str") in ["error", "failed", "cancelled", "failure"]:
                raise RuntimeError(f"ComfyUI workflow {prompt_id} encountered an problem: {status}")

        if prompt_id in history and "outputs" in history[prompt_id]:
            outputs = history[prompt_id]["outputs"]
            if outputs:
                return outputs
        time.sleep(poll_interval)
    raise TimeoutError(f"ComfyUI workflow did not complete within {timeout} seconds")


def patch_workflow(workflow_request: WorkflowRequest) -> dict:
    uuid_str = str(uuid.uuid4())
    remapped = copy.deepcopy(workflow_request.workflow)

    for patch in workflow_request.patches:
        target_node = None
        for _, node in remapped.items():
            if isinstance(node, dict) and node.get("_meta", {}).get("title") == patch.title:
                target_node = node
                break

        if target_node is None:
            raise ValueError(f"Patch title '{patch.title}' not found in workflow")

        inputs = target_node.get("inputs")
        if inputs is None:
            raise ValueError(f"Node '{patch.title}' has no inputs to patch")

        if patch.class_type == "LoadImage":
            uploaded_name = api_image_upload(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.png",
            )
            inputs["image"] = uploaded_name
        elif patch.class_type == "LoadVideo":
            uploaded_name = api_image_upload(
                patch.value,
                subfolder="api_inputs",
                filename=f"{uuid_str}_{patch.title}.mp4",
            )
            inputs["video"] = uploaded_name
        else:
            inputs["value"] = patch.value

    return remapped


def main(context: WorkflowContext) -> Image.Image:
    # ensure_comfy_alive()
    if not is_comfy_running():
        raise RuntimeError("ComfyUI is not running")

    # free aggessively for now
    api_free(unload_models=True, free_memory=False)

    workflow = patch_workflow(context.data)
    log_pretty("Remapped ComfyUI workflow", workflow)

    # Queue the workflow to ComfyUI
    queue_response = api_prompt(workflow)
    prompt_id = queue_response.get("prompt_id")
    if not prompt_id:
        raise ValueError("Failed to queue ComfyUI workflow")

    # Wait for the workflow to complete
    outputs = poll_until_resolved(prompt_id)

    # Find the first image in the outputs
    result = None
    for node_id, node_output in outputs.items():
        for output_name, output_data in node_output.items():
            if output_data and isinstance(output_data, list) and len(output_data) > 0:
                if "filename" in output_data[0] and "type" in output_data[0]:
                    if output_data[0]["type"] == "output":
                        # Get the generated image
                        filename = output_data[0]["filename"]
                        subfolder = output_data[0].get("subfolder", "")
                        result = api_view(filename, subfolder)
                        break
        if result:
            break

    # free aggessively for now
    api_free(unload_models=True, free_memory=False)

    if isinstance(result, Image.Image):
        return result

    raise ValueError("Image generation failed")

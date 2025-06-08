import base64
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
from io import BytesIO

from PIL import Image

from common.logger import log_pretty, logger
from images.schemas import ComfyWorkflow
from videos.schemas import ComfyWorkflow as ComfyWorkflowVideo

COMFY_PORT = 8188
COMFY_PATH = "/app/ComfyUI"
COMFY_PROCESS = None
COMFY_API_URL = f"http://127.0.0.1:{COMFY_PORT}"
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


def _read_and_log(pipe, logger_func, prefix):
    for line in iter(pipe.readline, b""):
        logger_func(f"[COMFY] {line.decode().rstrip()}")


def is_comfy_running(port=COMFY_PORT):
    """Check if ComfyUI is already listening on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_comfy():
    global COMFY_PROCESS
    if is_comfy_running():
        return

    logger.warning("Starting ComfyUI...")
    COMFY_PROCESS = subprocess.Popen(
        ["python", "main.py", "--disable-auto-launch", "--listen", "127.0.0.1", "--port", str(COMFY_PORT)],
        cwd=COMFY_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,  # We'll decode lines ourselves
        bufsize=1,  # Line buffered
    )

    # Start a thread to read stdout and log it
    threading.Thread(target=_read_and_log, args=(COMFY_PROCESS.stdout, logger.info, "COMFY"), daemon=True).start()

    # Wait for Comfy to be responsive
    for _ in range(300):  # Wait for up to 300 seconds
        if is_comfy_running():
            logger.info("✅ ComfyUI started successfully.")
            return
        time.sleep(1)

    raise RuntimeError("❌ ComfyUI failed to start after timeout.")


def ensure_comfy_alive():
    """Call this before each task."""
    if not is_comfy_running():
        start_comfy()


def queue_prompt(workflow):
    """Send a workflow to ComfyUI for processing."""
    p = {"prompt": workflow}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request(f"{COMFY_API_URL}/prompt", data=data)
    response = urllib.request.urlopen(req)
    return json.loads(response.read().decode("utf-8"))


def get_history(prompt_id):
    """Get the execution history for a specific prompt ID."""
    response = urllib.request.urlopen(f"{COMFY_API_URL}/history/{prompt_id}")
    return json.loads(response.read().decode("utf-8"))


# def get_image(filename, subfolder="", folder_type="output"):
#     """Get an image from ComfyUI's output directory."""
#     url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
#     response = urllib.request.urlopen(url)
#     img_data = response.read()
#     return Image.open(BytesIO(img_data))


def get_image(filename, subfolder="", folder_type="output"):
    """Get an image from ComfyUI's output directory."""
    # Construct the direct file path
    file_path = os.path.join(COMFY_PATH, folder_type, subfolder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file {file_path} does not exist.")

    return Image.open(file_path)


def get_video_path(filename, subfolder="", folder_type="output"):
    """Get a video from ComfyUI's output directory."""
    # Construct the direct file path
    file_path = os.path.join(COMFY_PATH, folder_type, subfolder, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file {file_path} does not exist.")

    return file_path


def wait_for_completion(prompt_id, timeout=300, check_interval=1):
    """Wait for the ComfyUI prompt to complete processing."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = get_history(prompt_id)
        if prompt_id in history and "status" in history[prompt_id]:
            status = history[prompt_id]["status"]
            log_pretty(f"Checking status for prompt {prompt_id}", status)
            if status.get("status_str") in ["error", "failed", "cancelled", "failure"]:
                raise RuntimeError(f"ComfyUI workflow {prompt_id} encountered an problem: {status}")

        if prompt_id in history and "outputs" in history[prompt_id]:
            outputs = history[prompt_id]["outputs"]
            if outputs:
                return outputs
        time.sleep(check_interval)
    raise TimeoutError(f"ComfyUI workflow did not complete within {timeout} seconds")


def free_resources(unload_models=True, free_memory=False):
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
        response = queue_prompt(dummy_workflow)
        prompt_id = response.get("prompt_id")
        if prompt_id:
            wait_for_completion(prompt_id, timeout=10, check_interval=1)
        else:
            logger.warning("Dummy cleanup prompt failed to queue.")
    except Exception as e:
        logger.warning(f"ComfyUI cleanup job failed: {e}")


def remap_workflow(workflow: ComfyWorkflow | ComfyWorkflowVideo, data) -> dict:
    """Remap the workflow to match ComfyUI's expected format.

    Args:
        workflow (dict): The ComfyUI workflow to remap
        data (ImageRequest): The request data containing parameter values

    Returns:
        dict: The remapped workflow with updated values
    """
    # Make a deep copy of the workflow to avoid modifying the original
    remapped = workflow.model_dump()

    # Iterate through all nodes
    for node_id, node in remapped.items():
        # Check if node has metadata with a title
        if "_meta" in node and "title" in node["_meta"]:
            title = node["_meta"]["title"]

            # Check if title starts with "api_"
            if title.startswith("api_"):
                param_name = title[4:]  # Remove "api_" prefix

                # Check if the parameter exists in the data object
                if hasattr(data, param_name):
                    param_value = getattr(data, param_name)

                    # Handle LoadImage type nodes specially
                    if node["class_type"] == "LoadImage" and param_value is not None:
                        # Convert base64 to image and save
                        temp_filename = f"temp_{param_name}_{int(time.time())}.png"

                        # Decode the base64 image
                        img_data = base64.b64decode(param_value)
                        img = Image.open(BytesIO(img_data))

                        # Save to ComfyUI's input directory
                        input_dir = os.path.join(COMFY_PATH, "input")
                        os.makedirs(input_dir, exist_ok=True)
                        img_path = os.path.join(input_dir, temp_filename)
                        img.save(img_path)

                        # Update the node's image input
                        node["inputs"]["image"] = temp_filename

                    # Handle all other node types dynamically
                    elif "inputs" in node and "value" in node["inputs"]:
                        # Only update if the parameter value is not None
                        if param_value is not None:
                            node["inputs"]["value"] = param_value

    return remapped

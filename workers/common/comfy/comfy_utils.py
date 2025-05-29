import base64
import json
import os
import socket
import subprocess
import time
import urllib.request
from io import BytesIO

from PIL import Image

from common.logger import logger

COMFY_PORT = 8188
COMFY_PATH = "/app/ComfyUI"
COMFY_PROCESS = None
COMFY_API_URL = f"http://127.0.0.1:{COMFY_PORT}"


def is_comfy_running(port=COMFY_PORT):
    """Check if ComfyUI is already listening on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_comfy():
    global COMFY_PROCESS
    if is_comfy_running():
        logger.info("✅ ComfyUI already running.")
        return

    logger.warning("Starting ComfyUI...")
    COMFY_PROCESS = subprocess.Popen(
        ["python", "main.py", "--disable-auto-launch", "--listen", "127.0.0.1", "--port", str(COMFY_PORT)],
        cwd=COMFY_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for Comfy to be responsive
    for _ in range(100):  # Wait for up to 100 seconds
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


def get_image(filename, subfolder="", folder_type="output"):
    """Get an image from ComfyUI's output directory."""
    url = f"{COMFY_API_URL}/view?filename={filename}&subfolder={subfolder}&type={folder_type}"
    response = urllib.request.urlopen(url)
    img_data = response.read()
    return Image.open(BytesIO(img_data))


def wait_for_completion(prompt_id, timeout=300, check_interval=1):
    """Wait for the ComfyUI prompt to complete processing."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        history = get_history(prompt_id)
        if prompt_id in history and "outputs" in history[prompt_id]:
            outputs = history[prompt_id]["outputs"]
            if outputs:
                return outputs
        time.sleep(check_interval)
    raise TimeoutError(f"ComfyUI workflow did not complete within {timeout} seconds")

import base64
import json
import time
import uuid
from typing import Any, Optional

import httpx
import websocket

from common.config import settings
from common.logger import logger, task_log


class ComfyClient:
    def __init__(self):
        self.server_url = settings.comfy_api_url
        if not self.server_url:
            raise RuntimeError("COMFY_API_URL environment variable is not set")

        self.http_client: Optional[httpx.Client] = None
        self.ws: Optional[websocket.WebSocket] = None
        self.ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://")

    def _get_http_client(self) -> httpx.Client:
        if not self.server_url:
            raise RuntimeError("COMFY_API_URL environment variable is not set")

        if self.http_client is None:
            self.http_client = httpx.Client(
                base_url=self.server_url,
                timeout=httpx.Timeout(60.0),
                follow_redirects=True,
            )
        return self.http_client

    def _connect_websocket(self) -> str:
        """Establish WebSocket connection, returns client_id."""
        client_id = str(uuid.uuid4())
        self.ws = websocket.WebSocket()
        self.ws.connect(f"{self.ws_url}/ws?clientId={client_id}")
        return client_id

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Helper to raise descriptive errors for ComfyUI responses."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            try:
                error_data = response.json()
                msg = json.dumps(error_data, indent=2)
                raise RuntimeError(f"ComfyUI Error ({response.status_code}):\n{msg}") from e
            except Exception:
                raise RuntimeError(f"ComfyUI Error ({response.status_code}): {response.text}") from e

    def queue_prompt(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Send a workflow to ComfyUI for processing."""
        client = self._get_http_client()
        response = client.post("/prompt", json={"prompt": workflow})
        self._raise_for_status(response)
        return response.json()

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Get the execution history for a specific prompt ID."""
        client = self._get_http_client()
        response = client.get(f"/history/{prompt_id}")
        self._raise_for_status(response)
        result = response.json()

        if not isinstance(result, dict):
            raise RuntimeError(f"ComfyUI history response is not a dict: {result}")

        if prompt_id in result:
            return result[prompt_id]

        return {}

    def get_completed_history(self, prompt_id: str) -> dict[str, Any]:
        """Get validated outputs from a completed workflow."""
        history = self.get_history(prompt_id)

        status = history.get("status", {})
        completed = status.get("completed", False)

        if not completed:
            raise RuntimeError(f"ComfyUI workflow {prompt_id} did not complete successfully: {history}")

        outputs = history.get("outputs")
        if not outputs:
            raise ValueError(f"No outputs found in history for prompt {prompt_id}")

        return outputs

    def upload_image(self, base64_str: str, subfolder: str, filename: str) -> str:
        """Upload an image / video to ComfyUI's input directory."""
        file_bytes = base64.b64decode(base64_str)
        files = {"image": (filename, file_bytes, "application/octet-stream")}
        data = {"subfolder": subfolder or "", "type": "input", "overwrite": "1"}

        client = self._get_http_client()
        response = client.post("/upload/image", files=files, data=data)
        self._raise_for_status(response)

        result = response.json()
        result_subfolder = result.get("subfolder")
        result_filename = result.get("name")

        if result_filename is None:
            raise RuntimeError(f"ComfyUI upload response missing filename: {result}")

        if result_subfolder:
            return f"{result_subfolder}/{result_filename}"
        return result_filename

    def download_file(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download output data from ComfyUI (image or video)."""
        client = self._get_http_client()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type, "channel": "raw"}
        response = client.get("/view", params=params)
        self._raise_for_status(response)
        return response.content

    def free_memory(self, unload_models: bool = True, free_memory: bool = False) -> None:
        """Trigger ComfyUI to release VRAM and/or unload models."""
        payload = {"unload_models": unload_models, "free_memory": free_memory}

        try:
            client = self._get_http_client()
            response = client.post("/free", json=payload)
            self._raise_for_status(response)
        except Exception as e:
            logger.warning(f"Failed to set resource cleanup flags: {e}")
            return

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
            prompt_response = self.queue_prompt(dummy_workflow)
            prompt_id = prompt_response.get("prompt_id")
            if not prompt_id:
                logger.warning("ComfyUI cleanup job failed: no prompt_id returned")
                return
        except Exception as e:
            logger.warning(f"ComfyUI cleanup job failed: {e}")

    def track_progress(self, prompt_id: str) -> None:
        """Track workflow progress via WebSocket until completion."""
        self._connect_websocket()

        if not self.ws:
            raise RuntimeError("WebSocket not connected")

        while True:
            try:
                out = self.ws.recv()
                if isinstance(out, bytes):
                    # ComfyUI sends binary data for previews (not needed for progress tracking)
                    logger.info("Received binary WebSocket message")
                    continue
                message = json.loads(out)
            except Exception as e:
                logger.error(f"Error receiving WebSocket message: {e}")
                raise

            msg_type = message.get("type")
            data = message.get("data", {})
            logger.debug(f"{msg_type}: {data}")

            match msg_type:
                case "executing":
                    node = data.get("node", "unknown")
                    task_log(f"Executing node: {node}")

                case "progress":
                    value = data.get("value", 0)
                    max_value = data.get("max", 100)
                    if value % 10 == 0 or value == max_value:
                        task_log(f"Progress: {value}/{max_value}")

                case "status":
                    # NOTE checking remaining queue to detect completion seemed to be reliable
                    status = data.get("status", {})
                    exec_info = status.get("exec_info", {})
                    queue_remaining = exec_info.get("queue_remaining")
                    if queue_remaining == 0:
                        task_log("Generation completed")
                        return

                case "executed":
                    # NOTE this does not allways get sent at end of workflow, so also check status messages
                    if data.get("prompt_id") == prompt_id:
                        task_log("Generation completed")
                        return

                case "execution_error":
                    if data.get("prompt_id") == prompt_id:
                        exception_message = data.get("exception_message", "Unknown error")
                        raise RuntimeError(f"ComfyUI workflow {prompt_id} failed: {exception_message}")

                case _:
                    logger.debug(f"Unhandled message type: {msg_type} - {str(data)[:250]}...")

    def close(self) -> None:
        """Close all connections and free memory."""
        try:
            self.free_memory(unload_models=True, free_memory=True)
        except Exception as e:
            logger.warning(f"Failed to free memory on close: {e}")

        if self.ws:
            self.ws.close()
            self.ws = None
        if self.http_client:
            self.http_client.close()
            self.http_client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

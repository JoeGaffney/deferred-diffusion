import os
import webbrowser

import hou


def open_docs():
    base_url = os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000")
    webbrowser.open(f"{base_url}/redoc")


def open_task(node):
    base_url = os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000")
    flower_url = base_url.replace("5000", "5555")

    task_id = node.parm("task_id").eval()
    if task_id:
        webbrowser.open(f"{flower_url}/task/{task_id}")
    else:
        webbrowser.open(f"{flower_url}")

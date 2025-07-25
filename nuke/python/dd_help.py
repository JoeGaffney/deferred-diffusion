import os
import webbrowser

import nuke


def open_docs():
    base_url = os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000")
    webbrowser.open(f"{base_url}/redoc")


def open_task():
    base_url = os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000")
    flower_url = base_url.replace("5000", "5555")

    node = nuke.thisNode()
    task_id = node["task_id"].value()
    if task_id:
        webbrowser.open(f"{flower_url}/task/{task_id}")
    else:
        nuke.message("No task ID set!")

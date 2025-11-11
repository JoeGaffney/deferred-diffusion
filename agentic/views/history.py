import json

import gradio as gr

from common.state import Deps


def _show_past_messages_json(past_messages: list):
    """Format past messages as JSON for display"""
    if not past_messages:
        return "No messages in history"
    return json.dumps(
        [msg.model_dump() if hasattr(msg, "model_dump") else str(msg) for msg in past_messages], indent=2
    )


def _show_deps_state(deps: Deps):
    """Format deps state for display"""
    if not deps or (not deps.images and not deps.videos):
        return "No media generated yet"

    state_info = {
        "total_images": len(deps.images),
        "total_videos": len(deps.videos),
        "pending_tasks": len(deps.get_pending_tasks()),
        "completed_media": len(deps.get_completed_media()),
        "images": [
            {
                "id": img.id,
                "status": img.status,
                "model": img.model,
                "prompt": img.prompt[:100] + "..." if len(img.prompt) > 100 else img.prompt,
                "local_file_path": img.local_file_path,
            }
            for img in deps.images
        ],
        "videos": [
            {
                "id": vid.id,
                "status": vid.status,
                "model": vid.model,
                "prompt": vid.prompt[:100] + "..." if len(vid.prompt) > 100 else vid.prompt,
                "local_file_path": vid.local_file_path,
            }
            for vid in deps.videos
        ],
    }

    return json.dumps(state_info, indent=2)


def create_history_component(past_messages_state, deps_object):
    """Create history component that accesses deps directly"""

    with gr.Accordion("History & State", open=False) as history_accordion:
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Messages"):
                    history_json = gr.Code(
                        value="No messages in history",
                        language="json",
                        label="Past Messages",
                        max_lines=20,
                    )

                with gr.Tab("Media State"):
                    deps_json = gr.Code(
                        value="No media generated yet",
                        language="json",
                        label="Session Media State",
                        max_lines=20,
                    )

    # Auto-update messages
    past_messages_state.change(
        fn=_show_past_messages_json,
        inputs=[past_messages_state],
        outputs=[history_json],
    )

    # Set up periodic refresh for deps (every 30 seconds, same as timer)
    deps_timer = gr.Timer(30)
    deps_timer.tick(fn=lambda: _show_deps_state(deps_object), outputs=[deps_json])

    return history_accordion, deps_json  # Return the deps display componen

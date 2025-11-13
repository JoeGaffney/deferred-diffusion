import gradio as gr

from common.state import Deps


def _show_past_messages_json(past_messages: list):
    """Format past messages as structured data for JSON display"""
    if not past_messages:
        return {"status": "No messages in history"}

    return {
        "message_count": len(past_messages),
        "messages": [msg.model_dump() if hasattr(msg, "model_dump") else str(msg) for msg in past_messages],
    }


def _show_deps_state(deps: Deps):
    """Format deps state as structured data for JSON display"""
    return {
        "summary": {
            "total_images": len(deps.images),
            "total_videos": len(deps.videos),
            "pending_videos": len(deps.get_pending_videos_ids()),
            "pending_images": len(deps.get_pending_images_ids()),
            "completed_media": len(deps.get_completed_media()),
        },
        "images": [img.model_dump() for img in deps.images],
        "videos": [vid.model_dump() for vid in deps.videos],
    }


def _get_media_gallery_items(deps: Deps):
    """Extract local file paths for media gallery display"""
    media_paths = []

    # Get images with local_file_path
    for img in deps.images:
        if hasattr(img, "local_file_path") and img.local_file_path:
            media_paths.append(img.local_file_path)

    # Get videos with local_file_path
    for vid in deps.videos:
        if hasattr(vid, "local_file_path") and vid.local_file_path:
            media_paths.append(vid.local_file_path)

    return media_paths


def create_history_component(past_messages_state: gr.State, deps_state: gr.State):
    """Create history component that accesses deps directly"""

    with gr.Accordion("History & State", open=False) as history_accordion:
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Messages"):
                    history_json = gr.JSON(
                        value={"status": "No messages in history"},
                        label="Past Messages",
                        show_label=True,
                    )

                with gr.Tab("Media State"):
                    deps_json = gr.JSON(
                        value={"status": "No media generated yet"},
                        label="Session Media State",
                        show_label=True,
                    )

                with gr.Tab("Media Gallery"):
                    media_gallery = gr.Gallery(
                        object_fit="contain",
                        columns=4,
                    )

    # Auto-updates
    past_messages_state.change(
        fn=_show_past_messages_json,
        inputs=[past_messages_state],
        outputs=[history_json],
    )

    deps_state.change(
        fn=_show_deps_state,
        inputs=[deps_state],
        outputs=[deps_json],
    )

    deps_state.change(
        fn=_get_media_gallery_items,
        inputs=[deps_state],
        outputs=[media_gallery],
    )

    return history_accordion, deps_json, media_gallery

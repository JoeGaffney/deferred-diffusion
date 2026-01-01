import gradio as gr


def _show_past_messages_json(past_messages: list):
    """Format past messages as structured data for JSON display"""
    if not past_messages:
        return {"status": "No messages in history"}

    return {
        "message_count": len(past_messages),
        "messages": [msg.model_dump() if hasattr(msg, "model_dump") else str(msg) for msg in past_messages],
    }


def create_history_component(past_messages_state: gr.State):
    """Create history component that accesses deps directly"""

    with gr.Accordion("History", open=False) as history_accordion:
        history_json = gr.JSON(
            value={"status": "No messages in history"},
            label="Past Messages",
            show_label=True,
        )

    # Auto-updates
    past_messages_state.change(
        fn=_show_past_messages_json,
        inputs=[past_messages_state],
        outputs=[history_json],
    )

    return history_accordion

import json

import gradio as gr


def _show_past_messages_json(past_messages: list):
    """Format past messages as JSON for display"""
    if not past_messages:
        return "No messages in history"
    return json.dumps(
        [msg.model_dump() if hasattr(msg, "model_dump") else str(msg) for msg in past_messages], indent=2
    )


def create_history_component(past_messages_state):
    """
    Create a complete history component with accordion.
    Returns the accordion component.
    Like a React component that manages its own state and events.
    """
    # Create the accordion (collapsible section)
    with gr.Accordion("Message History", open=False) as history_accordion:
        with gr.Column():
            history_json = gr.Code(
                value="No messages in history",  # Default placeholder
                language="json",
                label="Past Messages",
                max_lines=20,
            )

    # Auto-update when state changes
    past_messages_state.change(
        fn=_show_past_messages_json,
        inputs=[past_messages_state],
        outputs=[history_json],
    )

    return history_accordion

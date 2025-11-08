import json

import gradio as gr


def _show_past_messages_json(past_messages: list):
    """Format past messages as JSON for display in modal"""
    if not past_messages:
        return "No messages in history"
    return json.dumps(
        [msg.model_dump() if hasattr(msg, "model_dump") else str(msg) for msg in past_messages], indent=2
    )


def create_history_component(past_messages_state):
    """
    Create a complete history component with button and modal.
    Returns the button component - modal is handled internally.
    Like a React component that manages its own state and events.
    """
    # Create a state to track modal visibility
    modal_visible_state = gr.State(False)
    modal_visible_lablel_state = gr.State("Show History" if not modal_visible_state.value else "Hide History")
    # Create the button
    with gr.Column(scale=1, min_width=120):
        show_history_btn = gr.Button(modal_visible_lablel_state.value, variant="secondary")

    # Create the modal (hidden by default)
    with gr.Row(visible=False) as history_modal:
        with gr.Column():
            history_json = gr.Code(language="json", label="Past Messages")

    # Setup toggle event handler
    show_history_btn.click(
        lambda past_msgs, is_visible: (
            not is_visible,  # Update visibility state
            gr.update(visible=not is_visible),  # Toggle modal
            _show_past_messages_json(past_msgs) if not is_visible else "",  # Update content when showing
        ),
        inputs=[past_messages_state, modal_visible_state],
        outputs=[modal_visible_state, history_modal, history_json],
    )

    # Live updates every 5 seconds when modal is visible
    gr.Timer(2).tick(
        lambda past_msgs, is_visible: _show_past_messages_json(past_msgs) if is_visible else gr.skip(),
        inputs=[past_messages_state, modal_visible_state],
        outputs=[history_json],
    )

    return show_history_btn

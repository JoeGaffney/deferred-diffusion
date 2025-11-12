from __future__ import annotations as _annotations

import json

import gradio as gr
from pydantic import BaseModel
from pydantic_ai import ToolCallPart, ToolReturnPart

from agents.chat_agent import chat_agent
from agents.fetch_agent import fetch_agent
from common.logger import logger
from common.schemas import ToolCallCoupling
from common.state import Deps
from views.history import create_history_component


def get_media_content(content_dict: dict) -> gr.components.Component | None:
    """
    Checks for local file paths in the content_dict and appends appropriate media messages to the chatbot.
    Supports PNG images and MP4 videos.
    """
    local_file_path = None
    if isinstance(content_dict, dict):
        # Handle nested result structure
        if "result" in content_dict and isinstance(content_dict["result"], dict):
            local_file_path = content_dict["result"].get("local_file_path")
        else:
            local_file_path = content_dict.get("local_file_path")

    # Add image display if local file path exists and is an image
    if local_file_path and str(local_file_path).lower().endswith(".png"):
        return gr.Image(value=local_file_path, height="33%", width="33%")
    if local_file_path and str(local_file_path).lower().endswith(".mp4"):
        return gr.Video(value=local_file_path, height="33%", width="33%")
    return None


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list, deps: Deps):
    chatbot.append({"role": "user", "content": prompt})
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip(), gr.skip()

    async with chat_agent.run_mcp_servers():
        async with chat_agent.run_stream(prompt, deps=deps, message_history=past_messages) as result:
            tool_calls: dict[str, ToolCallCoupling] = {}  # Track calls by ID

            # First pass: collect all tool interactions
            for message in result.new_messages():
                for call in message.parts:
                    if isinstance(call, ToolCallPart):
                        if call.tool_call_id:
                            tool_calls[call.tool_call_id] = ToolCallCoupling(
                                tool_call_id=call.tool_call_id,
                                tool_call=call,
                            )
                    elif isinstance(call, ToolReturnPart):
                        if call.tool_call_id and call.tool_call_id in tool_calls:
                            tool_calls[call.tool_call_id].tool_return = call

            # Second pass: apply all completed tool calls together
            for coupling in tool_calls.values():
                if coupling.tool_call and coupling.tool_return:
                    # Add tool call message
                    metadata = {
                        "title": f"🛠️ Using {coupling.tool_call.tool_name}",
                        "id": coupling.tool_call_id or "",
                        # "status": "done",
                    }
                    call_args = coupling.tool_call.args_as_dict()
                    content = {"Arguments": call_args}

                    # Add tool return message
                    if isinstance(coupling.tool_return.content, BaseModel):
                        json_content = coupling.tool_return.content.model_dump_json()
                        content_dict = coupling.tool_return.content.model_dump()
                        content["Result"] = content_dict
                    else:
                        json_content = json.dumps(coupling.tool_return.content)
                        content_dict = (
                            coupling.tool_return.content if isinstance(coupling.tool_return.content, dict) else {}
                        )
                        content["Result"] = content_dict

                    chatbot.append(
                        {
                            "role": "assistant",
                            "content": gr.JSON(value=content),
                            "metadata": metadata,
                        }
                    )

                    media_content = get_media_content(content_dict)
                    if media_content:
                        media_message = {
                            "role": "assistant",
                            "content": media_content,
                        }
                        chatbot.append(media_message)

                    deps.add_or_update_media(content_dict, coupling.tool_return.tool_name or "")

            # Single yield after all tool interactions are processed
            if tool_calls:
                yield gr.skip(), chatbot, gr.skip(), gr.skip()

            # Finally, stream the main output
            chatbot.append({"role": "assistant", "content": ""})
            assistant_index = len(chatbot) - 1
            async for message in result.stream_text():
                chatbot[assistant_index]["content"] = message
                yield gr.skip(), chatbot, gr.skip(), gr.skip()

            logger.info(f"Stream Deps: {deps.images}, {deps.videos}")
            past_messages.extend(result.new_messages())
            yield gr.Textbox(interactive=True), gr.skip(), past_messages, deps.clone()


async def check_completed_generations(chatbot: list[dict], past_messages: list, deps: Deps):
    """Silently check for completed generations and add media to chat"""
    prompt = "fetch any incomplete generations that are in PENDING or STARTED state"
    logger.info(f"Checking for completed generations... {deps.images}, {deps.videos}")
    if not deps.get_pending_images_ids() and not deps.get_pending_videos_ids():
        return gr.skip(), gr.skip(), gr.skip()

    has_updates = False
    async with fetch_agent.run_mcp_servers():
        result = await fetch_agent.run(prompt, deps=deps, message_history=past_messages)
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolReturnPart):
                    if isinstance(call.content, BaseModel):
                        content_dict = call.content.model_dump()
                    else:
                        content_dict = call.content if isinstance(call.content, dict) else {}

                    # Only add media content - no tool call display
                    media_content = get_media_content(content_dict)
                    if media_content:
                        media_message = {
                            "role": "assistant",
                            "content": media_content,
                        }
                        chatbot.append(media_message)

                    deps.add_or_update_media(content_dict, call.tool_name or "")
                    has_updates = True

        if has_updates:
            # Update past_messages but don't show the tool calls in chat
            past_messages.extend(result.new_messages())
            if result.output:
                chatbot.append({"role": "assistant", "content": result.output or ""})

    # to avoid overwriting the UI if no updates
    return chatbot, past_messages, deps.clone()


async def handle_retry(chatbot, past_messages: list, deps: Deps, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages, deps):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value["text"]


with gr.Blocks(fill_height=True) as demo:
    past_messages = gr.State([], time_to_live=None)
    deps = gr.State(Deps(), time_to_live=None)

    with gr.Column(scale=10):
        chatbot = gr.Chatbot(
            label="AI Assistant",
            type="messages",
            examples=[
                {"text": "What are your toolset calls?"},
                {"text": "What image models are available?"},
                {"text": "What local video models are available?"},
            ],
            height="75vh",
        )
        prompt = gr.Textbox(
            lines=3,
            show_label=False,
            placeholder="What image models are available?",
            submit_btn=True,
        )
        # Create history component (includes button and modal with events)
        show_history_btn = create_history_component(past_messages, deps)

    generation = prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages, deps],
        outputs=[prompt, chatbot, past_messages, deps],
    )
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(handle_retry, [chatbot, past_messages, deps], [prompt, chatbot, past_messages, deps])
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])

    # Add timer for automatic checking
    timer = gr.Timer(30)  # 30 seconds
    # Timer event for checking completed generations
    timer.tick(
        check_completed_generations,
        inputs=[chatbot, past_messages, deps],
        outputs=[chatbot, past_messages, deps],
    )

if __name__ == "__main__":
    demo.launch()

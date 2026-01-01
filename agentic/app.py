from __future__ import annotations as _annotations

import json

import gradio as gr
from gradio.components.chatbot import MetadataDict
from gradio.processing_utils import PUBLIC_HOSTNAME_WHITELIST
from pydantic import BaseModel
from pydantic_ai import ToolCallPart, ToolReturnPart

# allow localhost in gradio images and videos
PUBLIC_HOSTNAME_WHITELIST.append("127.0.0.1")
PUBLIC_HOSTNAME_WHITELIST.append("localhost")

from agents.chat_agent import chat_agent
from agents.fetch_agent import fetch_agent
from common.logger import logger
from common.schemas import ToolCallCoupling
from common.state import Deps
from views.history import create_history_component


def get_media_content(content_dict: dict) -> gr.components.Component | None:
    """
    Checks for media URLs in the content_dict and appends appropriate media messages to the chatbot.
    Supports PNG images and MP4 videos.
    """
    urls = []
    if isinstance(content_dict, dict):
        # Handle nested result structure
        if "result" in content_dict and isinstance(content_dict["result"], dict):
            urls = content_dict["result"].get("output", [])
        else:
            urls = content_dict.get("output", [])

    # Add image display if url exists and is an image
    # For now, just display the first one
    if urls and isinstance(urls, list) and len(urls) > 0:
        url = urls[0]
        url_str = str(url).lower()
        if url_str.endswith(".png") or ".png?" in url_str:
            return gr.Image(value=url, height="33%", width="33%")
        if url_str.endswith(".mp4") or ".mp4?" in url_str:
            return gr.Video(value=url, height="33%", width="33%")
    return None


def get_content_dict_from_call(call: ToolReturnPart) -> dict:
    if isinstance(call.content, BaseModel):
        return call.content.model_dump()
    else:
        return call.content if isinstance(call.content, dict) else {}


async def run_agent_with_feedback(prompt: str, chatbot: list[gr.ChatMessage], past_messages: list, deps: Deps):
    chatbot.append(gr.ChatMessage(role="user", content=prompt))
    chatbot.append(gr.ChatMessage(role="assistant", content="🤔 Processing..."))
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip(), gr.skip()

    result = None
    try:
        async with chat_agent.run_mcp_servers():
            result = await chat_agent.run(prompt, deps=deps, message_history=past_messages)
    except Exception as e:
        chatbot.pop()  # Remove "thinking" message
        logger.error(f"Error: {e}")
        chatbot.append(gr.ChatMessage(role="assistant", content=f"Error: {str(e)}"))
        yield gr.Textbox(interactive=True), chatbot, gr.skip(), gr.skip()
        return

    chatbot.pop()  # Remove "thinking" message
    if result is None:
        chatbot.append(gr.ChatMessage(role="assistant", content="Error: No response from agent"))
        yield gr.Textbox(interactive=True), chatbot, gr.skip(), gr.skip()
        return

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
            metadata = MetadataDict(
                title=f"🛠️ Using {coupling.tool_call.tool_name}",
                id=coupling.tool_call_id or "",
                status="done",
            )
            call_args = coupling.tool_call.args_as_dict()
            content = {"Arguments": call_args}

            # Add tool return message
            content_dict = get_content_dict_from_call(coupling.tool_return)
            content["Result"] = content_dict

            chatbot.append(
                gr.ChatMessage(
                    role="assistant",
                    content=str(json.dumps(content, indent=2)),
                    metadata=metadata,
                )
            )

            media_content = get_media_content(content_dict)
            if media_content:
                chatbot.append(gr.ChatMessage(role="assistant", content=media_content))

            deps.add_or_update_media(content_dict, coupling.tool_return.tool_name or "")

    # Finally, render the agent's final output message
    if result.output:
        chatbot.append(gr.ChatMessage(role="assistant", content=result.output))

    yield gr.Textbox(interactive=True), chatbot, past_messages, deps.clone()


async def check_completed_generations(chatbot: list[gr.ChatMessage], past_messages: list, deps: Deps):
    """Silently check for completed generations and add media to chat"""
    prompt = "fetch any incomplete generations that are in PENDING or STARTED state"
    logger.info(f"Checking for completed generations... {deps.images}, {deps.videos}")
    if not deps.get_pending_images_ids() and not deps.get_pending_videos_ids():
        return gr.skip(), gr.skip(), gr.skip()

    has_updates = False
    result = None
    try:
        async with fetch_agent.run_mcp_servers():
            result = await fetch_agent.run(prompt, deps=deps, message_history=past_messages)

    except Exception as e:
        logger.error(f"Error during fetching completed generations: {e}")

    if result is None:
        return gr.skip(), gr.skip(), gr.skip()

    for message in result.new_messages():
        for call in message.parts:
            if isinstance(call, ToolReturnPart):
                content_dict = get_content_dict_from_call(call)

                # Only add media content - no tool call display
                media_content = get_media_content(content_dict)
                if media_content:
                    chatbot.append(gr.ChatMessage(role="assistant", content=media_content))
                deps.add_or_update_media(content_dict, call.tool_name or "")
                has_updates = True

    if has_updates:
        # Update past_messages but don't show the tool calls in chat
        past_messages.extend(result.new_messages())
        if result.output:
            chatbot.append(gr.ChatMessage(role="assistant", content=result.output))

        return chatbot, past_messages, deps.clone()

    # to avoid overwriting the UI if no updates
    return gr.skip(), gr.skip(), gr.skip()


async def handle_retry(chatbot, past_messages: list, deps: Deps, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in run_agent_with_feedback(previous_prompt, new_history, past_messages, deps):
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
        create_history_component(past_messages, deps)

    generation = prompt.submit(
        run_agent_with_feedback,
        inputs=[prompt, chatbot, past_messages, deps],
        outputs=[prompt, chatbot, past_messages, deps],
    )
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(handle_retry, [chatbot, past_messages, deps], [prompt, chatbot, past_messages, deps])
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])

    # Add timer for automatic checking
    timer = gr.Timer(30)  # 30 seconds
    timer.tick(
        check_completed_generations,
        inputs=[chatbot, past_messages, deps],
        outputs=[chatbot, past_messages, deps],
    )

if __name__ == "__main__":
    demo.launch()

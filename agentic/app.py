from __future__ import annotations as _annotations

import json

import gradio as gr
from pydantic import BaseModel
from pydantic_ai import ToolCallPart, ToolReturnPart

from agents.chat_agent import Deps, chat_agent
from views.history import create_history_component

deps = Deps(client=None)


def add_media_content(content_dict: dict, chatbot: list[dict]):
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
        media_message = {
            "role": "assistant",
            "content": gr.Image(value=local_file_path, height="33%", width="33%"),
        }
        chatbot.append(media_message)
    if local_file_path and str(local_file_path).lower().endswith(".mp4"):
        media_message = {
            "role": "assistant",
            "content": gr.Video(value=local_file_path, height="33%", width="33%"),
        }
        chatbot.append(media_message)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append({"role": "user", "content": prompt})
    yield gr.Textbox(interactive=False, value=""), chatbot, gr.skip()

    async with chat_agent.run_mcp_servers():
        async with chat_agent.run_stream(prompt, deps=deps, message_history=past_messages) as result:
            for message in result.new_messages():
                for call in message.parts:
                    if isinstance(call, ToolCallPart):
                        call_args = call.args_as_json_str()
                        metadata = {"title": f"🛠️ Using {call.tool_name}", "id": call.tool_call_id or ""}

                        gr_message = {
                            "role": "assistant",
                            "content": "Parameters: " + call_args,
                            "metadata": metadata,
                        }
                        chatbot.append(gr_message)
                    elif isinstance(call, ToolReturnPart):
                        id = call.tool_call_id or ""
                        for gr_message in chatbot:
                            valid = False
                            if gr_message and gr_message.get("metadata") != None:
                                valid = True

                            if valid:
                                if gr_message.get("metadata", {}).get("id", "") == id:
                                    if isinstance(call.content, BaseModel):
                                        json_content = call.content.model_dump_json()
                                        content_dict = call.content.model_dump()
                                    else:
                                        json_content = json.dumps(call.content)
                                        content_dict = call.content if isinstance(call.content, dict) else {}
                                    gr_message["content"] += f"\nOutput: {json_content}"
                                    add_media_content(content_dict, chatbot)

                    # must yield after each tool call part to stream properly ??
                    yield gr.skip(), chatbot, gr.skip()

            chatbot.append({"role": "assistant", "content": ""})
            async for message in result.stream_text():
                chatbot[-1]["content"] = message
                yield gr.skip(), chatbot, gr.skip()

            past_messages.extend(result.new_messages())
            yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def check_completed_generations(past_messages: list, chatbot: list[dict]):
    """Silently check for completed generations and add media to chat"""
    prompt = "fetch any incomplete generations that are in PENDING or STARTED state"
    print("Checking for completed generations...")
    async with chat_agent.run_mcp_servers():
        result = await chat_agent.run(prompt, deps=deps, message_history=past_messages)
        valid_message = []
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolReturnPart):
                    if isinstance(call.content, BaseModel):
                        content_dict = call.content.model_dump()
                    else:
                        content_dict = call.content if isinstance(call.content, dict) else {}

                    # Only add media content - no tool call display
                    add_media_content(content_dict, chatbot)
                    valid_message.append(message)

        # Update past_messages but don't show the tool calls in chat
        past_messages.extend(result.new_messages())

    return chatbot, past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]["content"]
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]["content"], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value["text"]


with gr.Blocks() as demo:
    past_messages = gr.State([])

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
        show_history_btn = create_history_component(past_messages)

    generation = prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages],
        outputs=[prompt, chatbot, past_messages],
    )
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages])
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])

    # Add timer for automatic checking
    timer = gr.Timer(30)  # 30 seconds
    # Timer event for checking completed generations
    timer.tick(check_completed_generations, inputs=[past_messages, chatbot], outputs=[chatbot, past_messages])

if __name__ == "__main__":
    demo.launch()

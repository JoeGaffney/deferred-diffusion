# Agentic Layer

The **agentic layer** is a **local, interactive orchestration environment** for AI-driven workflows.
It uses **Pydantic AI**, **OpenAI agents**, and connects to the **Deferred Diffusion MCP server** for image and video generation.
It is **local-first**, supports **multi-step, multi-modal workflows**, and maintains **conversation state within the app**.

## **Directory Structure**

```
/agentic          # Interactive agentic orchestration layer
├── generated/    # generated client
│── /agents       # AI agents implementing reasoning and workflow logic
│   ├── chat_agent.py           # General purpose
│   ├── sequence_agent.py       # Handles scene/shot sequencing
│   └── ...                     # Additional agents as needed
│── /tools        # Wrappers for calling /api or MCP configuration
│── /views        # Gradio UI components and interaction logic
│── /schemas      # Pydantic schemas
│── /utils        # Helper functions
│── /common       # Shared modules: logging, settings, session persistence
│── app.py        # Gradio entrypoint managing conversation state and UI
```

## **Typical Workflow**

1. User enters a prompt in the Gradio app.
2. Agent decides on actions (e.g., generate image, refine sequence) and calls **tools**.
3. Tools call MCP endpoints for model inference.
4. Results are returned to agent and stored in **local session state**.
5. Gradio UI displays outputs and allows the user to refine prompts for multi-turn interactions.

## Dev notes

base64 encoded strings are massive and fill up context we wrap these at the call level and convert to and from file paths.

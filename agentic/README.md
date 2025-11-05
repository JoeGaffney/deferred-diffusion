## **WIP Design for Agentic Layer**

The **agentic layer** is a **local, interactive orchestration environment** for AI-driven workflows.
It uses **Pydantic AI**, **OpenAI agents**, and connects to the **Deferred Diffusion MCP server** for image and video generation. Or the API directly with the generated client.
It is **local-first**, supports **multi-step, multi-modal workflows**, and maintains **conversation state within the app**.

### **Directory Structure**

```
/agentic          # Interactive agentic orchestration layer
├── generated/    # generated client
│── /agents       # AI agents implementing reasoning and workflow logic
│   ├── sequence_agent.py       # Handles scene/shot sequencing
│   ├── storyboard_agent.py     # Handles full storyboard orchestration
│   └── ...                     # Additional agents as needed
│── /tools        # Wrappers for calling /api or MCP endpoints
│── /views        # Gradio UI components and interaction logic
│── /schemas      # Pydantic schemas, can import from /api for consistency
│── /utils        # Helper functions, prompt formatting, embeddings
│── /common       # Shared modules: logging, settings, session persistence
│── app.py        # Gradio entrypoint managing conversation state and UI
```

### **Key Conventions**

- **Agents** implement multi-step reasoning and call tools as needed.
- **Tools** wrap `/api` or MCP calls, returning typed Pydantic responses.
- **Schemas** ensure typed and validated communication between the agentic layer and `/api` endpoints.
- **Views / app.py** handle **all state management**, multi-turn conversation memory, and display of results (images, sequences, etc.).
- **Session state** is local; user prompts, refinements, and generated outputs are maintained within the app.

### **Typical Workflow**

1. User enters a prompt in the Gradio app.
2. Agent parses prompt using Pydantic schemas.
3. Agent decides on actions (e.g., generate image, refine sequence) and calls **tools**.
4. Tools call `/api` or MCP endpoints for model inference.
5. Results are returned to agent and stored in **local session state**.
6. Gradio UI displays outputs and allows the user to refine prompts for multi-turn interactions.

# DEV notes.

```
    # MCP client will be here Your auto-generated MCP server is now available at https://app.base.url/mcp.
    base_url = os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000")

    # Client connection from generated
    return AuthenticatedClient(
        base_url=os.getenv("DDIFFUSION_API_ADDRESS", "http://127.0.0.1:5000"),
        token=os.getenv("DDIFFUSION_API_KEY", ""),
        raise_on_unexpected_status=True,
    )
```

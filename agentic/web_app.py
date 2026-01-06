import uvicorn

from agents.chat_agent import chat_agent

# pydantic AI web app instance
app = chat_agent.to_web()

# Run the app with any ASGI server:
# uvicorn my_module:app --host 127.0.0.1 --port 7932
# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    uvicorn.run("web_app:app", host="0.0.0.0", port=7932, reload=True, reload_dirs=["api"])

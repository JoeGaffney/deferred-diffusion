from fastapi import FastAPI

from api_v2.router import router

app = FastAPI(title="Deferred Diffusion V2 API", description="Task-driven architecture", version="2.0.0")

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check():
    return {"status": "ok"}
    return {"status": "ok"}
    return {"status": "ok"}
    return {"status": "ok"}
    return {"status": "ok"}

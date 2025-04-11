import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from agentic import router as agentic
from image import router as image
from text import router as text
from utils import device_info
from utils.utils import free_gpu_memory
from video import router as video

app = FastAPI(title="API")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500, content={"message": "Internal server error", "detail": str(exc), "path": request.url.path}
    )


@app.middleware("http")
async def cleanup_gpu_memory(request: Request, call_next):
    free_gpu_memory()
    response = await call_next(request)
    return response


app.include_router(image.router, prefix="/api")
app.include_router(text.router, prefix="/api")
app.include_router(video.router, prefix="/api")
app.include_router(agentic.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to the API!"}


# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    # NOTE need to run single-threaded so we don't run out of Vram with multiple requests
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

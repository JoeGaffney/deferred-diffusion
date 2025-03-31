import uvicorn
from fastapi import FastAPI
from utils import device_info

app = FastAPI(title="API")

from image import router as image
from text import router as text
from video import router as video

app.include_router(image.router, prefix="/api")
app.include_router(text.router, prefix="/api")
app.include_router(video.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to the API!"}


# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    # NOTE need to run single-threaded so we don't run out of Vram with multiple requests
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)

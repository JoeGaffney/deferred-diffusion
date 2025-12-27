import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from admin import router as admin
from common.logger import logger
from images import router as images
from texts import router as texts
from utils.utils import truncate_strings
from videos import router as videos
from workflows import router as workflows

# NOTE imporant keep name API as clients will use the title
app = FastAPI(title="API")


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    cleaned = []
    for err in exc.errors():
        d = dict(err)
        cleaned.append(
            {
                "loc": d.get("loc", []),
                "msg": d.get("msg", ""),
                "type": d.get("type", ""),
            }
        )

    logger.warning(f"Validation error on {request.url.path}: {cleaned}")
    return JSONResponse(status_code=422, content={"detail": cleaned})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": truncate_strings(str(exc), 1000),
            "path": request.url.path,
        },
    )


app.include_router(images.router, prefix="/api")
app.include_router(texts.router, prefix="/api")
app.include_router(videos.router, prefix="/api")
app.include_router(workflows.router, prefix="/api")
app.include_router(admin.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to the API!"}


# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=["api"], workers=1)

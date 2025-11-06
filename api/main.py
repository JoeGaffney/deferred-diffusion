import json
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from common.logger import logger
from images import router as images
from texts import router as texts
from videos import router as videos


def truncate_strings(data: Any, max_length: int = 100) -> Any:
    if isinstance(data, dict):
        return {k: truncate_strings(v, max_length) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_strings(item, max_length) for item in data]
    elif isinstance(data, str):
        return data if len(data) <= max_length else data[:max_length] + "..."
    else:
        return data


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
async def log_requests(request: Request, call_next):
    if request.method == "POST":
        body_bytes = await request.body()
        request._body = body_bytes

        body_str = ""
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
            truncated_body = truncate_strings(body_json)
            body_str = json.dumps(truncated_body)
        except Exception as e:
            # If not valid JSON, log first 100 chars
            body_str = body_bytes.decode("utf-8", errors="replace")
            body_str = body_str[:100] + "..." if len(body_str) > 100 else body_str

        logger.info(f"[Middleware] {request.method} {request.url.path} body: {body_str}")
    else:
        logger.info(f"[Middleware] {request.method} {request.url.path}")

    # Continue processing
    response = await call_next(request)
    return response


app.include_router(images.router, prefix="/api")
app.include_router(texts.router, prefix="/api")
app.include_router(videos.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "Welcome to the API!"}


# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=["api"], workers=1)

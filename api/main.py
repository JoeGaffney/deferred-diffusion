import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastmcp import FastMCP

from admin import router as admin
from common.config import settings
from common.logger import logger
from files import router as files
from images import router as images
from texts import router as texts
from utils.utils import truncate_strings
from videos import router as videos
from workflows import router as workflows

# NOTE imporant keep name API as clients will use the title
fastapi_app = FastAPI(title="API")


@fastapi_app.exception_handler(RequestValidationError)
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


@fastapi_app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "detail": truncate_strings(str(exc), 1000),
            "path": request.url.path,
        },
    )


fastapi_app.include_router(images.router, prefix="/api")
fastapi_app.include_router(texts.router, prefix="/api")
fastapi_app.include_router(videos.router, prefix="/api")
fastapi_app.include_router(workflows.router, prefix="/api")
fastapi_app.include_router(files.router, prefix="/api")
fastapi_app.include_router(admin.router, prefix="/api")


@fastapi_app.get("/")
def root():
    return RedirectResponse(url="/docs")


@fastapi_app.get("/health")
def health():
    return {"status": "healthy"}


# Combine mcp and fastapi
if settings.enable_mcp:
    mcp = FastMCP.from_fastapi(
        app=fastapi_app,
        name="MCP",
        httpx_client_kwargs={
            "headers": {
                "Authorization": "Bearer secret-token",
            }
        },
    )

    mcp_app = mcp.http_app(path="/mcp", transport="streamable-http", stateless_http=True)
    app = FastAPI(
        routes=[
            *fastapi_app.routes,
            *mcp_app.routes,
        ],
        lifespan=mcp_app.lifespan,
    )
else:
    app = fastapi_app

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Run Uvicorn programmatically for convenience
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, reload_dirs=["api"], workers=1)

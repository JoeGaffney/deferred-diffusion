# Deferred Diffusion - Architecture V2 Plan

## Overview: Task-Driven Architecture

The original V1 architecture grouped logic by **domain** (`images`, `texts`, `videos`). As the system grew, models became heavily multi-modal (image-to-text, text-to-video, etc.), rendering strict domain boundaries obsolete.

V2 transitions the stack to a **Task-Driven Architecture**. Instead of specific routers and schemas (e.g., `ImageWorkerResponse`), the system treats all inference as a generic `Task`, accepting a unified input schema based on the model and always returning standard output arrays (`output_files`, `output_text`, etc.).

**Key Architectural Shifts:**

1. **Unified Return Structure**: Every worker task returns generic arrays instead of domain-specific keys.
2. **Build-Time Separation**: API and Worker environments remain completely isolated via Docker build processes. However, both containers will now share a common repository of `tasks/` where `schemas` and `logic` live together but are imported selectively.
3. **Leave ComfyUI Support for the time being**: Ignore the `workflows/` module and ComfyUI sidecar have been dropped to focus purely on programmatic, code-defined API endpoints.
4. **Simplified Clients**: Operations like Houdini/Nuke plugins no longer need to build complex endpoint routers; they simply dispatch a payload to `/tasks` and poll for a generic payload containing `output_files`.

---

## Directory Structure

```text
deferred-diffusion/
├── api_v2/               # 🟢 Lightweight API Server (FastAPI)
│   ├── main.py           # Single set of generic routes for tasks
|   ├── router.py         # use the actually router still make one
│   ├── dependencies.py   # whats this?
│   └── requirements.txt  # FastAPI, Pydantic, Celery (NO ML LIBRARIES)
│
├── workers_v2/           # 🔴 Heavy Execution Engine (Celery)
│   ├── worker.py         # Registers Celery tasks from the `tasks` module
│   ├── context.py        # Business logic, loading models/files
│   └── requirements.txt  # PyTorch, Diffusers, Transformers, etc.
│
├── tasks/                # 🟡 The Source of Truth (Task Definitions)
│   ├── __init__.py
│   ├── common_schemas.py # The base TaskRequest / TaskResponse schemas
│   ├── flux_1/
│   │   ├── schemas.py    # Pydantic input schema for Flux 1 (Safe for API) used also by worker
│   │   └── execute.py    # PyTorch/ML execution pipeline (Safe ONLY for Worker) at can import stuff in workers module
│   ├── wan_2_1/
│   │   ├── schemas.py
│   │   └── execute.py
│   └── gpt_4_1-mini/
│       ├── schemas.py
│       └── execute.py
│
├── clients/              # 🟣 Generated Clients & Integrations
│   ├── houdini/
│   ├── nuke/
│   └── it_tests/
│
├── docker-compose.yml    # Main stack (API, Worker, Redis)
└── Makefile              # Unified build tools
```

---

## The "Pod" Pattern (Tasks Directory)

The core principle of V2 is the `tasks/` folder, which acts as a collection of self-contained "**Pods**".

Each pod (e.g., `tasks/flux_1/`) contains:

- `schemas.py`: Defines the input parameters required by the model. It **ONLY** imports Pydantic and standard libraries. The `api/` container dynamically loads these to build the OpenAPI spec.
- `execute.py`: Contains the actual pipeline logic. It imports heavy ML libraries. The `api/` container **NEVER** imports this file.

### Unified Schemas (`tasks/common_schemas.py`)

All tasks now share a single output structure:

```python
from pydantic import BaseModel
from typing import List, Optional

class TaskResponse(BaseModel):
    output_files: List[str] = []  # Paths/URLs to images, videos, audio, etc.
    output_text: List[str] = []   # LLM text, captions, etc.
    logs: List[str] = []          # Execution logs and warnings
    status: str                   # 'pending', 'processing', 'completed', 'failed'
```

---

## API & Routing

V1 had dozens of endpoints: `POST /images/create`, `POST /videos/create`, etc.
V2 simplifies the API surface massively.

### `POST /api/tasks`

Accepts a generic task payload. The API uses a discriminated union of all task schemas (built dynamically from `tasks/*/schemas.py`) to validate the payload.

- **Payload**: `{ "task_name": "flux_1", "params": { "prompt": "...", "seed": 42 } }`
- **Response**: `202 Accepted` with `{ "task_id": "uuid" }`

### `GET /api/tasks/{task_id}`

Returns the current status of the task. If completed, it contains the populated `TaskResponse`.

- **Behavior**: The API intercepts the `output_files` array from the Celery worker and promotes internal file paths (e.g., `/tmp/result.mp4`) into securely signed, accessible short-lived URLs before returning it to the client.

---

## Docker Build Strategy

Because `tasks/` contains both API-safe and Worker-only files, the separation is enforced entirely by dependencies and imports at runtime:

1. **API Image (`Dockerfile.api`)**:
   - Copies `/api/` and `/tasks/` into the container.
   - Installs **only** `api/requirements.txt` (FastAPI, Uvicorn, Pydantic).
   - Scans `/tasks/*/schemas.py` to register Pydantic schemas dynamically.

2. **Worker Image (`Dockerfile.workers`)**:
   - Copies `/workers/` and `/tasks/` into the container.
   - Installs `workers/requirements.txt` (PyTorch, Diffusers, etc.).
   - Scans `/tasks/*/execute.py` to register Celery tasks.

This cleanly eliminates the need for the `make copy-schemas` script, keeping the logic modular without bloating the API backend.

---

## Client Integration Workflows

For tools like **Houdini** or **Nuke**, integrating V2 is highly optimized:

1. The host tool formats its parameters into base64 strings and numerical values matching the specific schema for the task type.
2. It sends `POST /api/tasks`.
3. It polls `GET /api/tasks/{task_id}` for completion.
4. Once marked `completed`, the client iterates over the `output_files` array and downloads each file into the respective Node application, regardless of whether it's an image, a video, or an audio track.

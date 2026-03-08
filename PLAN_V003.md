# Deferred Diffusion - Architecture V3 Plan

## Overview: Physically Segmented Task-Driven Architecture

The V2 architecture consolidated tasks but placed them in a unified `tasks/` directory, relying on queue routing to prevent lightweight workers from evaluating heavy ML code.

V3 takes the **Task-Driven Architecture** principle further by strictly **physically dividing the tasks by their hardware requirements**. This guarantees no cross-contamination between dependencies while retaining dynamic routing.

**Key Architectural Shifts:**

1. **Unified Return Structure**: Every worker task returns `TaskResponse` (generic arrays like `output_files`, `output_text`).
2. **Physical Execution Split**: Tasks are grouped strictly into `local_worker/tasks` (for heavy GPU requirements) and `external_worker/tasks` (for lightweight API bounds).
3. **Dynamic Pathing Framework**: The API routes dynamically map to the physical location of the task. For example, the path becomes `/api/tasks/external_worker/gpt_4_1_mini`.
4. **Shared Schemas**: Instead of putting the base schema in a specific worker context, `TaskResponse` and `TaskStatus` live in a top-level `shared/` directory accessible by both the API and all workers.

---

## Directory Structure

```text
deferred-diffusion/
├── api_v2/                 # 🟢 Lightweight API Server (FastAPI)
│   ├── main.py             # Entry point
│   ├── router.py           # Dynamically scans worker directories to build routes
│   └── requirements.txt    # FastAPI, Pydantic, Celery, Redis
│
├── shared/                 # 🟡 Cross-Boundary Types
│   ├── __init__.py
│   └── common_schemas.py   # TaskResponse, TaskStatus
│
├── local_worker/           # 🔴 Heavy Execution Engine (GPU / Local)
│   ├── worker.py           # Registers Celery tasks from local_worker/tasks/*
│   ├── context.py          # GPU/Local business logic, model caching
│   ├── requirements.txt    # PyTorch, Diffusers, Transformers
│   └── tasks/              # GPU Task Definitions
│       ├── flux_1/
│       │   ├── schemas.py  # Pydantic input params
│       │   └── execute.py  # PyTorch model logic
│       └── wan_2_1/
│           ├── schemas.py
│           └── execute.py
│
├── external_worker/        # 🔵 Light Execution Engine (CPU / APIs)
│   ├── worker.py           # Registers Celery tasks from external_worker/tasks/*
│   ├── context.py          # API keys, session management
│   ├── requirements.txt    # Requests, OpenAI, Celery
│   └── tasks/              # CPU Task Definitions
│       └── gpt_4_1_mini/
│           ├── schemas.py  # Pydantic input params
│           └── execute.py  # External API request logic
│
├── clients/                # 🟣 Generated Clients & Integrations (Houdini/Nuke)
├── docker-compose.yml      # Main stack configuration
├── Dockerfile.api_v2       # Builds API container
├── Dockerfile.worker.local # Builds GPU container
└── Dockerfile.worker.ext   # Builds CPU container
```

---

## API & Routing

V3 builds REST endpoints that perfectly map the filesystem layout. It scans the `*_worker/tasks/` folders when building the OpenAPI spec.

### `POST /api/tasks/{worker_type}/{task_name}`

For instance, invoking `POST /api/tasks/external_worker/gpt_4_1_mini`.
The `router.py` automatically loops through `['local_worker', 'external_worker']` to build these exact routes.

- **Routing and Queues**:
  - Instead of guessing what queue to send to, the API sends it to a queue named identically to the `worker_type` (e.g., `queue="local_worker"` or `queue="external_worker"`).
  - The tasks are named `{worker_type}.{task_name}` inside Celery.

### `GET /api/tasks/{task_id}`

Same behavior as V2. Retrieves `TaskResponse` via UUID.

---

## Safety and Scalability

By structuring the project this way:

1. **Separation of Concerns**: The API container simply scans the schemas, but the execution modules are entirely physically separate.
2. **Crash Prevention**: An external worker image will literally never contain a `flux_1` folder, making it impossible to accidentally import `torch`.
3. **Optimized Build Contexts**: `Dockerfile.worker.local` can COPY only the `local_worker/` directory, drastically reducing container size for the CPU worker.

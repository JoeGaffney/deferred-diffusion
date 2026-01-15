# Deferred Diffusion AI Coding Instructions

## Big Picture Architecture

- **Feature-Based Structure**: Code is grouped by domain (`images`, `texts`, `videos`) in both `api/` and `workers/`.
- **API (FastAPI)**: Handles requests, validates input, and queues tasks via Celery. Entry point: `api/main.py`.
- **Workers (Celery)**: Executes inference. Split into `gpu` (local models) and `cpu` (external APIs) queues. Entry point: `workers/worker.py`.
- **Data Flow**: Client -> API (POST) -> Redis (Broker) -> Worker -> Redis -> API (GET) -> Client.
- **Statelessness**: Workers are stateless; results are returned as file storage paths and "promoted" to signed URLs by the API.
- Everything is ran though the docker containers defined in `docker-compose.yml`.

## Critical Workflows

- **Schema Syncing**: `api/<feature>/schemas.py` is the source of truth. **ALWAYS** run `make copy-schemas` after editing API schemas to sync them to `workers/`.
- **Client Generation**: Run `make all` to update typed clients in `clients/` (Houdini, Nuke, it_tests) after API changes.
- **Testing**:
  - Worker tests: `make test-worker TEST_PATH=images/local/test_flux_1.py`
  - Integration tests: `make test-it-tests`

## Coding Patterns & Conventions

- **Adding a New Model**:
  1. Update `api/<feature>/schemas.py` (add to `ModelName` enum and `MODEL_META`).
  2. Run `make copy-schemas`.
  3. Create `workers/<feature>/local/<model_name>.py` with a `main(context)` function.
  4. Register task in `workers/<feature>/tasks.py` using `@typed_task` and **lazy imports** inside the function.
- **Worker Tasks**:
  - Use `validate_request_and_context(args)` to initialize a `Context` object.
  - `Context` (`workers/<feature>/context.py`) handles image loading, resizing, and seed management.
  - Return `ImageWorkerResponse(base64_data=..., logs=get_task_logs())`.
- **API Routers**:
  - Use `operation_id` in FastAPI decorators (e.g., `operation_id="images_create"`) for consistent client generation.
  - Use `promote_result_to_storage` in GET endpoints to convert worker Base64 results to signed URLs.
- **ComfyUI Sidecar**:
  - Runs as a separate service (`docker-compose.comfy.yml`) on port 8188.
  - API: `/api/workflows/router.py` routes to the `comfy` Celery queue.
  - Workflows use ComfyUI's **API JSON** format (not standard UI JSON).
  - Use `Patch` objects in `WorkflowRequest` to dynamically swap node values (e.g., prompts, seeds) in the workflow.
- **VFX Clients (Houdini/Nuke)**:
  - **Threaded Execution**: **ALWAYS** use `@threaded` decorators for API calls to prevent UI freezing.
  - **UI Updates**: Use host-specific callbacks (e.g., `hou.ui.postEventCallback`) to update node status/info from background threads.
  - **Data Flow**: Convert node inputs to Base64 using `input_to_base64` (Houdini) or `node_to_base64` (Nuke) before sending to API.
  - **Generated Clients**: Do not edit files in `generated/` folders; they are overwritten by `make generate-clients`.

## Key Files

- `api/common/schemas.py`: Shared task statuses and Base64 types.
- `workers/common/config.py`: Worker-specific settings (storage, model paths).
- `Makefile`: Central hub for builds, syncing, and testing.
- `docker-compose.yml`: Defines all services (API, workers, Redis, ComfyUI).
- README.md: overall project documentation.

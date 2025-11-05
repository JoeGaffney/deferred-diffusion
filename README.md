# deferred-diffusion

**Deferred Diffusion** is a **self-hosted, scalable AI inference stack** with a fully **typed, testable API**. It supports **local GPU models** and can route tasks to **external AI services** such as Replicate, OpenAI, or RunwayML. The system is **containerized**, automatically downloads all models and dependencies, and is **stateless**, allowing tasks to run across multiple workers without relying on local file paths. This makes deployments **predictable, cross-platform, and easy to scale**.

It provides a **modular API and worker architecture** built with **FastAPI** and **Celery**, letting local models and external providers run seamlessly in the same system. The API queues tasks through a message broker, and worker services pick them up for processing. Workers can execute:

- **Local ML pipelines** using the Python ecosystem (e.g., diffusers, PyTorch)
- **External inference tasks** via APIs such as Replicate, OpenAI, and RunwayML

An **intelligent model cache** keeps the last-used local model resident in GPU memory for fast reuse.
**Text encoders** are run on the CPU, and **prompt embeddings** are cached to maximize available VRAM during inference.

Clients interact with the API through clean typed REST endpoints, with a built-in **Swagger UI** for testing and inspection.

Example **Houdini** and **Nuke** clients are included to demonstrate integration into node-based VFX pipelines.

## **Project Structure Overview**

This project follows a **feature-based structure**, grouping related components together by domain (`images`, `texts`, `videos`). This approach ensures a clear separation of concerns and improves maintainability, scalability, and collaboration.

We try to use plural to adhere to REST best practices.

### **Cohesion & Readability**

- All components related to a specific AI task (`images`, `texts`, `videos`) are grouped together.
- They are grouped in a sense of what main data type they return, but can have multi model inputs.
  - eg. images can accept image and text inputs but always returns image based data.
- Eliminates the need to navigate across multiple directories to understand a feature.
- New developers can quickly locate relevant code without confusion.

### **Scalability for AI Projects**

- AI models often require **domain-specific logic**. Keeping `schemas.py`, `context.py`, and `models/` in the same module makes it easier to extend functionality.
- If a new AI domain (`audio`, `3D`, etc.) is introduced, the structure remains consistent just duplicate the existing pattern.

```
/api
│── /images # Grouped by results type
│ ├── schemas.py # ✅ Pydantic schemas (data validation)
│ ├── router.py # ✅ API routes (FastAPI) Calls worker tasks
│── /texts
│ ├── ...
│── /videos
│ ├── ...
│── /agentic # Agentic area is a bit experimental
│ ├── agents/
│ ├── schemas.py
│ ├── context.py
│ ├── router.py
│── /common # ✅ Shared components
│── /utils # ✅ General-purpose utilities (helpers, formatters, etc.)
│── /tests # ✅ Tests mirror the /api structure
│── main.py # ✅ FastAPI entry point
│── worker.py # ✅ Celery
│── pytest.ini # ✅ Test configuration
```

```
/workers
│── /images # Grouped by results type
│ ├── models/ # ✅ AI models (ML/DL models, weights, configs)
│ ├── external_models/ # ✅ external AI models
│ ├── schemas.py # ✅ Pydantic schemas (data validation mirrors from API)
│ ├── context.py # ✅ Business logic layer
│ ├── tasks.py # ✅ Celery tasks route to models
│── /texts
│ ├── ...
│── /videos
│ ├── ...
│── /common # ✅ Shared components
│── /utils # ✅ General-purpose utilities (helpers, formatters, etc.)
│── /tests # ✅ Tests mirror the /workers structure
│── worker.py # ✅ Celery
│── pytest.ini # ✅ Test configuration
```

```
/clients
│── /it_tests
│ ├── generated/ # generated client
│ ├── tests/
│── /houdini
│ ├── python/generated/ # generated client
│── /nuke
│ ├── python/generated/ # generated client
│── openapi.json # API spec
```

## Model naming / pathing

User-facing model choices are simple names like "flux-1" or "flux-1-pro". The actual model calls and implementations are defined in the worker pipeline. Worker tasks follow these user-driven names but may share common logic for variants.

For example, "flux-1" might internally use:

- "black-forest-labs/FLUX.1-Krea-dev"
- "black-forest-labs/FLUX.1-Kontext-dev"
- "black-forest-labs/FLUX.1-Fill-dev"

Depending on the inputs (e.g., whether an image is provided), we internally route to the most appropriate model variant.

We avoid cluttering user model choices with minor versions (.1, .2, etc.) and instead select the best available minor version. This approach allows us to properly test and verify model behaviors for both external and local models without requiring users to understand implementation details.

The model pipelines themselves serve as the source of truth for what models are actually used. This is especially important given various optimizations and edge cases that may apply.

#### Model Registration Philosophy

Model definitions are **version-controlled in code**, not loaded dynamically from configuration files.

This design choice ensures:

- **Full test coverage** and deterministic behavior across releases
- **Stable API contracts** between `/api` and `/workers`
- **Clear traceability** between user-facing model identifiers and their actual implementations

Developers who want to extend or modify available models can do so by editing the typed definitions directly in code:

- `api/images/schemas.py`
- `workers/images/tasks.py` or `workers/images/models/`

Each new model entry should include:

1. A Pydantic schema entry in `ModelNameLocal` or `ModelNameExternal`
2. A corresponding task or pipeline implementation
3. Updated tests under `tests/images`

This deliberate coupling between **model definitions, pipelines, and tests** is what makes `deferred-diffusion` reliable and reproducible for self-hosted AI inference.

#### **Flow Diagram Concept**

```
Client API Request
    │
    ▼
ImageRequest / VideoRequest schema
    │
    ▼
API ModelName (user-facing choice, e.g., "flux-1")
    ├─ Sends a Celery task to the workers
    └─ Selects the worker queue (CPU or GPU) via task_queue property
    │
    ▼
Worker / Task Router
    ├─ Confirms queue assignment (CPU/GPU)
    └─ Selects the appropriate pipeline function based on ModelName (lazy import)
    │
    ▼
ImageContext / VideoContext
    ├─ Initializes inputs: images, mask, seed, width/height
    ├─ Initializes adapters & controlnets if enabled
    └─ Provides helper functions:
          - get_generation_mode()
          - ensure_divisible()
          - cleanup()
    │
    ▼
Pipeline Function (pure function)
    ├─ Calls one of:
          - text_to_image_call(context)
          - image_to_image_call(context)
          - inpainting_call(context)
    ├─ Internally selects the exact model(s) / transformer variants:
          - Flux: Krea / Kontext / Fill
          - WAN, VEO variants based on context
    ├─ Applies adapters and controlnets as needed
    └─ Prepares prompt embeddings for inference
    │
    ▼
Pipeline Execution
    ├─ Runs inference on GPU or CPU
    └─ Produces output: PIL Image (or video frames for VideoContext)
    │
    ▼
Context.save_image() / Context.save_video()
    └─ Writes temporary file path for output
    │
    ▼
Worker / API Response
    └─ Encodes output as base64 (VideoWorkerResponse / ImageWorkerResponse)
```

## Building

Run primarily in the docker containers because of the multi service workflows and the multi copies of model downloads.

```bash
make all
```

### Local setup Windows

For local venv mainly to get intellisense on the packages and some local testing.

```bash
./start_venv_setup.bat
```

## Testing

Pytest is used for integration tests confirming the models run.

You can call from the make file.

```bash
make test-worker
make test-it-tests
```

See the make file for more info.

## Releasing

Tag and push to github will trigger github actions to the do the release.

- Currently there is no testing in the CI because of the gpu compute nature of things so please run the test suite locally on main before any releases or merges.

To make a local release.

```bash
make create-release
make tag-and-push
```

### Deploying the Release on a Server

1. **Change into the directory** containing the `docker-compose.yml` file.

2. **Ensure Docker Desktop is installed** on the server.

3. **Pull and run the containers**:

   ```bash
   docker-compose down
   docker-compose pull
   docker-compose up -d --no-build
   ```

### Requirements

- **Storage**: An NVMe drive with **at least 500GB** of available space is recommended.
- **GPU** Nvidia GPU with at least 12gb VRAM. 24 GB recommended.
- **Environment Variables**: Ensure all required environment variables are set on the host.

#### Required Environment Variables

Server for the containers

```env
OPENAI_API_KEY=your-openai-key # For OpenAI services
RUNWAYML_API_SECRET=your-runway-secret # For RunwayML services
REPLICATE_API_TOKEN=your-replicate-token # For Replicate API access
HF_TOKEN=your-huggingface-token # For Hugging Face model access
DDIFFUSION_API_KEYS=Welcome1!,Welcome2! # API keys for authentication
```

For the clients where the tool sets are used

```env
DDIFFUSION_API_ADDRESS=http://127.0.0.1:5000 # API server address
DDIFFUSION_API_KEY=Welcome1! # API key for client authentication
```

### Testing workers

Tests are included inside the containers these can be ran to verify and also to download any missing models.

```bash
docker-compose exec gpu-workers pytest tests/images/models/test_flux.py -vs
docker-compose exec gpu-workers pytest tests/images/models -vs
docker-compose exec gpu-workers pytest tests/texts/models -vs
docker-compose exec gpu-workers pytest tests/videos/models -vs
```

The external tests are split out.

```bash
docker-compose exec gpu-workers pytest tests/images/external_models -vs
docker-compose exec gpu-workers pytest tests/texts/external_models -vs
docker-compose exec gpu-workers pytest tests/videos/external_models -vs
```

## Clients

These are examples on how to simply get things on the path you could use rez or any other way preferred way to get the modules and plugins loaded.

Adjust directories depending on where you have the folders and the versions of your application. Examples are given for a windows environment.

### Houdini Setup

#### Python Modules

The following need to be available to Houdini for the API client and agents to work.

- httpx

You can install like this to put on roaming path.

```bash
"C:\Program Files\Side Effects Software\Houdini 20.5\bin\hython.exe" -m pip install httpx
```

#### Env file

```env
HOUDINI_PATH = C:/development/deferred-diffusion/clients/houdini;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/clients/houdini;&
PYTHONPATH = C:/development/deferred-diffusion/clients/houdini/python;&
```

### Nuke plug-in setup

#### Python modules

The following need to be available to Nuke for the API client to work.

- httpx
- attrs

You can install like this.

```bash
"C:\Program Files\Nuke14.0\python.exe" -m pip install httpx attrs
```

#### Adding to the path

Update your

- C:\Users\USERNAME\.nuke\init.py

```python
import nuke

# Centralized Nuke plugin path (your custom directory)
custom_plugin_path = r"C:\development\deferred-diffusion\clients\nuke"

# Add your custom plugin paths
nuke.pluginAddPath(custom_plugin_path)
print(f"Custom plugin paths from {custom_plugin_path} have been added.")
```

## MISC

### Docker helpers

To optimize volumes and virtual disk useful after model deletions

#### Kill Docker Desktop and related processes

```bash
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "com.docker.*" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "vmmemWSL" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wslhost" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "wsl" -Force -ErrorAction SilentlyContinue
```

```bash
Optimize-VHD -Path "Y:\DOCKER\DockerDesktopWSL\disk\docker_data.vhdx" -Mode Full
```

# deferred-diffusion

Multi model API that can run diffusion and other models with py-torch and external services.

The API will push tasks to worker broker and workers will pick this up. Workers can run process tasks using python ML ecosystem, external tasks which call ML providers.

Client will call the API get end points to check for task completion. See swagger ui for more info.

Currently example Houdini HDA's are provided as it already provides a rich compositing node based ui, but would be possible to add more applications or a standalone ui.

## **Project Structure Overview**

This project follows a **feature-based structure**, grouping related components together by domain (`images`, `texts`, `videos`). This approach ensures a clear separation of concerns and improves maintainability, scalability, and collaboration.

We try to use plural to adhere to REST best practices.

### **📂 Why This Structure?**

#### ✅ **Cohesion & Readability**

- All components related to a specific AI task (`images`, `texts`, `videos`) are grouped together.
- They are grouped in a sense of what main data type they return, but can have multi model inputs.
  - eg. images can accept image and text inputs but always returns image based data.
- Eliminates the need to navigate across multiple directories to understand a feature.
- New developers can quickly locate relevant code without confusion.

#### ✅ **Scalability for AI Projects**

- AI models often require **domain-specific logic**. Keeping `schemas.py`, `context.py`, and `models/` in the same module makes it easier to extend functionality.
- If a new AI domain (`audio`, `3D`, etc.) is introduced, the structure remains consistent just duplicate the existing pattern.

```
/api
│── /images # Grouped by results type
│ ├── schemas.py # ✅ Pydantic schemas (data validation)
│ ├── router.py # ✅ API routes (FastAPI) Calls worker tasks
│── /texts
│ ├── schemas.py
│ ├── router.py
│── /videos
│ ├── schemas.py
│ ├── router.py
│── /agentic
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
| |── api_schemas.py # symlink ?
│ ├── schemas.py # ✅ Pydantic schemas (data validation)
│ ├── context.py # ✅ Business logic layer
│ ├── tasks.py # ✅ Celery task
│── /texts
│ ├── models/
│ ├── external_models/ # ✅ external AI models
│ ├── schemas.py
│ ├── context.py
│ ├── tasks.py
│── /videos
│ ├── models/
│ ├── external_models/ # ✅ external AI models
│ ├── schemas.py
│ ├── context.py
│ ├── tasks.py
│── /common # ✅ Shared components
│── /utils # ✅ General-purpose utilities (helpers, formatters, etc.)
│── /tests # ✅ Tests mirror the /workers structure
│── worker.py # ✅ Celery
│── pytest.ini # ✅ Test configuration
```

Agentic area is a bit experimental; the agents can call on other modules, for example, calling the "texts" or "images" models for vision processing by the use of tools.

## Building

Run primarily in the docker containers because of the multi service workflows and the multi copies of model downloads.

```bash
make all
```

### Local setup Windows

For local venv

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

## Releasing

will make /releases/deferred-diffusion-alpha
Which will have a tar of the images setup

```bash
make create-release
```

Tag & push docker images to the hub - optional

```bash
make tag-and-push
```

### Deploy the release on a server

Docker desktop is required and run

```bash
docker-compose down
docker load -i deferred-diffusion-api.tar
docker load -i deferred-diffusion-workers.tar
docker-compose up -d --no-build
```

An NVME drive with min 500gb of space is potentially required and env vars need to be configured on the host.

### Required Environment Variables

Server for the containers

```env
OPENAI_API_KEY=your-openai-key # For OpenAI services
RUNWAYML_API_SECRET=your-runway-secret # For RunwayML services
REPLICATE_API_TOKEN=your-replicate-token # For Replicate API access
HF_TOKEN=your-huggingface-token # For Hugging Face model access
DDIFFUSION_API_KEYS=Welcome1! # API keys for authentication
```

For the clients where the toolsets are used

```env
DDIFFUSION_API_ADDRESS=http://127.0.0.1:5000 # API server address
DDIFFUSION_API_KEY=Welcome1! # API key for client authentication
```

### Testing workers

Tests are included inside the containers these can be ran to verify and also to download any missing models.

```bash
docker-compose exec gpu-workers pytest tests/images -vs
docker-compose exec gpu-workers pytest tests/texts -vs
docker-compose exec gpu-workers pytest tests/videos -vs
```

## Toolsets

These are examples on how to simply get things on the path you could use rez or any other way preferred way to get the modules and plugins loaded.

Adjust directories depending on where you have the toolset folders and the versions of your application. Examples are given for a windows environment.

### HDA's houdini setup

#### Python Modules

The following need to be available to Houdini for the API client and agents to work.

- httpx

You can install like this to put on roaming path.

```bash
"C:\Program Files\Side Effects Software\Houdini 20.5\bin\hython.exe" -m pip install httpx
```

#### Env file

```env
HOUDINI_PATH = C:/development/deferred-diffusion/hda;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/hda;&
PYTHONPATH = C:/development/deferred-diffusion/hda/python;&
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
custom_plugin_path = r"C:\development\deferred-diffusion\nuke"

# Add your custom plugin paths
nuke.pluginAddPath(custom_plugin_path)
print(f"Custom plugin paths from {custom_plugin_path} have been added.")
```

## MISC

### Docker helpers

To optimize volumes and virtual disk useful after model deletions

```bash
Optimize-VHD -Path "Y:\DOCKER\DockerDesktopWSL\disk\docker_data.vhdx" -Mode Full
```

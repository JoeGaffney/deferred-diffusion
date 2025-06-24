# deferred-diffusion

Multi model API that can run diffusion and other models with py-torch and external services. This can be ran locally or on another machine on the same network accessing the same paths.

Currently example Houdini HDA's are provided as it already provides a rich compositing node based ui, but would be possible to add more applications or a standalone ui.

The api will push tasks to worker broker and workers will pick this up. Some endpoints will async wait for tasks some extra long ones will require end client to re-poll and check progress. Workers can run process tasks using python ML ecosystem, external tasks which call ML providers.

# **Project Structure Overview**

This project follows a **feature-based structure**, grouping related components together by domain (`images`, `texts`, `videos`). This approach ensures a clear separation of concerns and improves maintainability, scalability, and collaboration.

We try to use plural to adhere to REST best practices.

## **📂 Why This Structure?**

### ✅ **Cohesion & Readability**

- All components related to a specific AI task (`images`, `texts`, `videos`) are grouped together.
- They are grouped in a sense of what main data type they return, but can have multi model inputs.
  - eg. images can accept image and text inputs but always returns image based data.
- Eliminates the need to navigate across multiple directories to understand a feature.
- New developers can quickly locate relevant code without confusion.

### ✅ **Scalability for AI Projects**

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

### Toolsets (example)

```
/hda
│── /python # Grouped by results type
│ ├── /generated # generated api client
│ ├── /api/
│ ├── ├── api_image_node.py # node and api calling logic
│ ├── utils.py
│ ├── config.py
│── cop_image_node.hda
│── cop_text_node.hda
│── cop_video.hda
```

# Local setup Windows

```sh
./start_venv_setup.bat
```

# Building

Run primarly in the docker containers because of the multi service worflows.

Make all

# Testing

Pytest is used for integration tests confirming the models run.

You can call from the make file.

- Make test-images
- Make test-texts

Or locally

```
cd api
pytest -vs
```

# Docker

- docker-compose build
- docker-compose up

To optimize volumes and virtual disk useful after model deletions

- Optimize-VHD -Path "Y:\DOCKER\DockerDesktopWSL\disk\docker_data.vhdx" -Mode Full

Combined

- docker-compose up --build

Tag & push

- docker tag deferred-diffusion-api:latest joegaffney/deferred-diffusion:latest
- docker push joegaffney/deferred-diffusion:latest

# Toolsets

These are examples on how to simply get things on the path you could use rez or any other way preferred way to get the modules and plugins loaded.

Adjust directories depending on where you have the toolset folders and the versions of your application. Examples are given for a windows environment.

## HDA's houdini setup

## Python Modules

The following need to be available to houdini for the api client and agents to work.

- httpx

You can install like this to put on roaming path.

```

"C:\Program Files\Side Effects Software\Houdini 20.5\bin\hython.exe" -m pip install httpx

```

## Env file

```

HOUDINI_PATH = C:/development/deferred-diffusion/hda;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/hda;&
PYTHONPATH = C:/development/deferred-diffusion/hda/python;&

```

## Nuke plug-in setup

### Python modules

The following need to be available to nuke for the api client and agents to work.

- httpx
- attrs

You can install like this.

```

"C:\Program Files\Nuke14.0\python.exe" -m pip install httpx attrs

```

### Adding to the path

Update your

- C:\Users\USERNAME\.nuke\init.py

```

import nuke

nuke.message("Nuke initialized!")

# Centralized Nuke plugin path (your custom directory)

custom_plugin_path = r"C:\development\deferred-diffusion\nuke"

# Add your custom plugin paths

nuke.pluginAddPath(custom_plugin_path)

# Test message (useful for debugging)

print(f"Custom plugin paths from {custom_plugin_path} have been added.")

```

```

```

# deferred-diffusion

deferred-diffusion api that can run diffusion and other models with py-torch. This can be ran locally or on another machine on the same network accessing the same paths.

Currently Houdini HDA's are provided as it already provides a rich compositing node based ui, but would be possible to add more applications or a standalone ui.

# **Project Structure Overview**

This project follows a **feature-based structure**, grouping related components together by domain (`image`, `text`, `video`). This approach ensures a clear separation of concerns and improves maintainability, scalability, and collaboration.

## **📂 Why This Structure?**

### ✅ **Cohesion & Readability**

- All components related to a specific AI task (`image`, `text`, `video`) are grouped together.
- Eliminates the need to navigate across multiple directories to understand a feature.
- New developers can quickly locate relevant code without confusion.

### ✅ **Scalability for AI Projects**

- AI models often require **domain-specific logic**. Keeping `schemas.py`, `context.py`, and `models/` in the same module makes it easier to extend functionality.
- If a new AI domain (`audio`, `3D`, etc.) is introduced, the structure remains consistent—just duplicate the existing pattern.

### ✅ **Easier Debugging & Maintenance**

- If an issue occurs in the `image` module, it's contained within `image/`, reducing debugging time.
- Reduces the risk of modifying shared logic that could unintentionally affect other domains.

### ✅ **Faster Development**

- Different teams can work independently on `image`, `text`, and `video` without interfering with each other.
- Encourages modular development, making it easier to test and iterate on individual features.

This structure balances **clarity, maintainability, and scalability**, making it well-suited for AI-driven projects where different domains have distinct processing needs. 🚀

/api
│── /image # Grouped by results type
│ ├── models/ # ✅ AI models (ML/DL models, weights, configs)
│ ├── schemas.py # ✅ Pydantic schemas (data validation)
│ ├── context.py # ✅ Business logic layer
│ ├── router.py # ✅ API routes (FastAPI)
│── /text
│ ├── models/
│ ├── schemas.py
│ ├── context.py
│ ├── router.py
│── /video
│ ├── models/
│ ├── schemas.py
│ ├── context.py
│ ├── router.py
│── /common # ✅ Shared components
│── /utils # ✅ General-purpose utilities (helpers, formatters, etc.)
│── /tests # ✅ Tests mirror the /api structure
│── main.py # ✅ FastAPI entry point
│── pytest.ini # ✅ Test configuration

# Setup windows

```sh
./start_venv_setup.bat
```

# To run main

```sh
./start_dev.bat
```

# Testing

Pytest is used for integration tests confirming the models run.

```
cd api
pytest -v
pytest .\tests\text\models\test_qwen_2_5_vl_instruct.py
```

## To test running models directly

A main method can be added

```
cd api/

python -m image.models.stable_diffusion_3_5
python -m video.models.ltx_video
python -m video.models.stable_video_diffusion
```

# HDA's houdini setup

## Python Modules

httpx needs to be available to houdini for the api client to work.

You can install like this to put on roaming path.

```
"C:\Program Files\Side Effects Software\Houdini XX.X\bin\hython.exe" -m pip install httpx
"C:\Program Files\Side Effects Software\Houdini 20.5\bin\hython.exe" -m pip install httpx
```

## Env file

Adjust directories depending on where you have the hda folder.

```
HOUDINI_PATH = C:/development/deferred-diffusion/hda;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/hda;&
#not strictly required as hda add the Python folder on load
#PYTHONPATH = C:/development/deferred-diffusion/hda/Python;&
```

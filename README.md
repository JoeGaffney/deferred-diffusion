# deferred-diffusion

deferred-diffusion api that can run diffusion and other models with py-torch. This can be ran locally or on another machine on the same network accessing the same paths.

A common context is used to simplify the passing of data from ui toolkits.

Endpoints will be split based on the main type or result eg. image, video, process and text.

Currently Houdini HDA's are provided as it already provides a rich compositing node based ui, but would be possible to add
more applications or a standalone ui.

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

```
cd api/

python -m image.models.stable_diffusion_3_5
python -m video.models.ltx_video
python -m video.models.stable_video_diffusion
```

# HDA's houdini setup

## Python Modules

httpx needs to be available to houdini fothe api client to work.

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

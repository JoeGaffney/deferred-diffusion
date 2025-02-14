# deferred-diffusion

deferred-diffusion

# to run main

./start_dev.bat

# to test running models directly

cd api/

python -m img_to_img.models.stable_diffusion_xl_refine
python -m img_to_img.models.stable_diffusion_xl_inpainting
python -m img_to_video.models.ltx_video
python -m img_to_video.models.stable_video_diffusion

# add to houdini

HOUDINI_PATH = C:/development/deferred-diffusion/hda;&
HOUDINI_OTLSCAN_PATH = C:/development/deferred-diffusion/hda;&
#PYTHONPATH = C:/development/deferred-diffusion/hda/Python;&

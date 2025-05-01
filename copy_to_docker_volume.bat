@echo off
SETLOCAL

SET "VOLUME_NAME=deferred-diffusion_hf_cache"
SET "BASE_SOURCE=Y:\HF_HOME\hub"
SET "BASE_TARGET=/hub"

echo Starting model transfer to Docker volume: %VOLUME_NAME%

REM Copy models one by one
call :CopyModelDir ".locks"
call :CopyModelDir "models--XLabs-AI--flux-controlnet-canny-v3"
call :CopyModelDir "models--black-forest-labs--FLUX.1-schnell"
call :CopyModelDir "models--city96--FLUX.1-schnell-gguf"
@REM call :CopyModelDir "models--city96--HunyuanVideo-I2V-gguf"
@REM call :CopyModelDir "models--city96--Wan2.1-I2V-14B-480P-gguf"
call :CopyModelDir "models--depth-anything--Depth-Anything-V2-Large-hf"
call :CopyModelDir "models--diffusers--controlnet-canny-sdxl-1.0-small"
call :CopyModelDir "models--diffusers--controlnet-depth-sdxl-1.0-small"
@REM call :CopyModelDir "models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1"
call :CopyModelDir "models--facebook--sam-vit-base"
call :CopyModelDir "models--fluently--Fluently-XL-v4"
call :CopyModelDir "models--h94--IP-Adapter"
@REM call :CopyModelDir "models--hunyuanvideo-community--HunyuanVideo"
@REM call :CopyModelDir "models--hunyuanvideo-community--HunyuanVideo-I2V"
@REM call :CopyModelDir "models--IDEA-Research--grounding-dino-base"
@REM call :CopyModelDir "models--InstantX--SD3-Controlnet-Canny"
@REM call :CopyModelDir "models--Lightricks--LTX-Video-0.9.5"
@REM call :CopyModelDir "models--llava-hf--llava-1.5-7b-hf"
@REM call :CopyModelDir "models--Qwen--Qwen2.5-VL-3B-Instruct"
call :CopyModelDir "models--SG161222--RealVisXL_V4.0"
@REM call :CopyModelDir "models--stabilityai--stable-diffusion-3.5-medium"
@REM call :CopyModelDir "models--stabilityai--stable-diffusion-3-medium-diffusers"
@REM call :CopyModelDir "models--stabilityai--stable-diffusion-x4-upscaler"
@REM call :CopyModelDir "models--stabilityai--stable-diffusion-xl-base-1.0"
@REM call :CopyModelDir "models--stabilityai--stable-diffusion-xl-refiner-1.0"
@REM call :CopyModelDir "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
@REM call :CopyModelDir "models--THUDM--CogVideoX1.5-5b-I2V"
@REM call :CopyModelDir "models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers"

echo All models have been copied successfully.
goto :EOF

:CopyModelDir
SET "DIR_NAME=%~1"
SET "SOURCE=%BASE_SOURCE%\%DIR_NAME%"
SET "TARGET=%BASE_TARGET%/%DIR_NAME%"

echo Copying: %DIR_NAME%
docker run --rm ^
    -v %VOLUME_NAME%:/HF_HOME ^
    -v "%SOURCE%":/model_src:ro ^
    ubuntu bash -c "mkdir -p /HF_HOME%TARGET% && cp -a /model_src/. /HF_HOME%TARGET%/"
exit /b 0

ENDLOCAL
@echo off
SETLOCAL

SET "VOLUME_NAME=deferred-diffusion_hf_cache"
SET "SOURCE=Y:\HF_HOME"

echo Starting transfer of HF_HOME contents to root of Docker volume: %VOLUME_NAME%

docker run --rm ^
    -v %VOLUME_NAME%:/HF_HOME ^
    -v "%SOURCE%":/source_data:ro ^
    ubuntu bash -c "cp -a /source_data/. /HF_HOME/"

echo Transfer completed.

ENDLOCAL
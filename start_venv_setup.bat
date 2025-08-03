@echo off
REM Check if the virtual environment directory exists
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
CALL venv\Scripts\activate

REM Check if the virtual environment is activated
IF "%VIRTUAL_ENV%"=="" (
    echo Failed to activate virtual environment. Aborting.
    exit /b 1
)

echo Upgrading pip...
python.exe -m pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

echo Installing requirements...
pip install --upgrade -r api/requirements.txt 
pip install --upgrade -r workers/requirements.txt 
pip install --upgrade -r infra/requirements.txt 

echo Installing diffusers from GitHub...
pip install -U git+https://github.com/huggingface/diffusers.git@main --no-deps

echo Installing lang SAM...
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git --no-deps
pip install sam2

pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git

echo Virtual environment activated and requirements installed.
CALL venv\Scripts\activate
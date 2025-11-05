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
pip uninstall diffusers
pip install --upgrade -r api/requirements.txt 
pip install --upgrade -r workers/requirements.txt 
pip install --upgrade -r agentic/requirements.txt 
pip install -r workers/requirements_no_deps.txt --no-deps

echo installing nunchaku...
pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.1/nunchaku-1.0.1+torch2.6-cp312-cp312-win_amd64.whl --no-deps

echo Virtual environment activated and requirements installed.
CALL venv\Scripts\activate
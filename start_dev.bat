
taskkill /F /IM python.exe

CALL venv\Scripts\activate

START python api/main.py 

timeout /t 45 /nobreak

Call openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path hda/Python/generated --overwrite
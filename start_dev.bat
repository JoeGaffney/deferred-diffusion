
taskkill /F /IM python.exe

CALL venv\Scripts\activate

START python api/main.py 

timeout /t 20 /nobreak

Call openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path hda/Python/api_client --overwrite
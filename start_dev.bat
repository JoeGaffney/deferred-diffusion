
taskkill /F /IM python.exe

CALL venv\Scripts\activate

START python api/main.py 

timeout /t 15 /nobreak

REM should possibly make a module for this
Call openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path hda/python/generated --overwrite
Call openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path nuke/python/generated --overwrite
Call openapi-python-client generate --url http://127.0.0.1:5000/openapi.json --output-path it_tests/generated --overwrite
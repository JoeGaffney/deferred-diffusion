taskkill /IM cypress.exe /F
taskkill /IM node.exe /F
taskkill /F /IM python.exe

CALL venv\Scripts\activate

START python api/main.py 

cd webapp/
npm run dev
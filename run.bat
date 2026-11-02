@echo off
echo ====================================================
echo             STARTING SMARTCART AI PLATFORM          
echo    Backend:  http://localhost:8000                  
echo    Docs:     http://localhost:8000/docs             
echo    Frontend: http://localhost:5173                  
echo    Admin:    admin@smartcart.ai / admin123          
echo ====================================================

start "SmartCart AI Backend" cmd /k "cd backend && .\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
start "SmartCart AI Frontend" cmd /k "cd frontend && npm run dev"

echo Services launched in separate windows. Press any key to exit this launcher.
pause


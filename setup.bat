@echo off
REM ============================================================================
REM SmartCart AI — Automated Setup Script for Windows
REM ============================================================================
echo [SmartCart AI] Initializing automated setup...

cd /d "%~dp0"

echo [1/4] Setting up Python virtual environment...
if not exist "backend\venv" (
    python -m venv backend\venv
)
call backend\venv\Scripts\activate.bat

echo [2/4] Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r backend\requirements.txt

echo [3/4] Initializing SQLite database, seeding catalog, and setting up warm snapshots...
python backend\app\seed.py
python backend\scripts\import_dummyjson.py
python backend\scripts\retailrocket\train_warm_start.py

echo [4/4] Installing frontend npm dependencies...
cd frontend
call npm install
cd ..

echo ============================================================================
echo [SmartCart AI] Setup completed successfully!
echo To run backend:  cd backend ^& ..\backend\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
echo To run frontend: cd frontend ^& npm run dev
echo To run tests:    .\backend\venv\Scripts\python.exe -m pytest backend\tests -q
echo ============================================================================

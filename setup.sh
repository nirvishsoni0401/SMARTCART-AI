#!/usr/bin/env bash
# ==============================================================================
# SmartCart AI — Automated Setup Script for POSIX (Linux / macOS)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[SmartCart AI] Initializing automated setup..."

echo "[1/4] Setting up Python virtual environment..."
if [ ! -d "backend/venv" ]; then
    python3 -m venv backend/venv
fi
source backend/venv/bin/activate

echo "[2/4] Installing backend dependencies..."
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

echo "[3/4] Initializing SQLite database and seeding catalog..."
python backend/app/seed.py

echo "[4/4] Installing frontend npm dependencies..."
cd frontend
npm install
cd ..

echo "=============================================================================="
echo "[SmartCart AI] Setup completed successfully!"
echo "To run backend:  source backend/venv/bin/activate && uvicorn app.main:app --port 8000"
echo "To run frontend: cd frontend && npm run dev"
echo "To run tests:    source backend/venv/bin/activate && pytest backend/tests -q"
echo "=============================================================================="

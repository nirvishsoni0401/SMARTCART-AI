"""
SmartCart AI — Phase 10D Offline Report Generator (Root wrapper)
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from scripts.retailrocket.generate_offline_report import run_all_generators

if __name__ == "__main__":
    run_all_generators()


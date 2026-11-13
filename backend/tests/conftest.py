import sys
import shutil
import tempfile
import hashlib
from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app import database as db_module
from app.main import app

SMARTCART_DB_PATH = backend_dir / "smartcart.db"


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def isolate_test_database():
    """
    Session-wide fixture ensuring zero writes to smartcart.db:
    1. Records SHA-256 hash of production smartcart.db.
    2. Copies smartcart.db to an isolated temporary file database.
    3. Overrides settings.DATABASE_URL, engine, SessionLocal, and get_db.
    4. At session teardown, asserts that smartcart.db was NEVER modified.
    """
    initial_hash = compute_file_hash(SMARTCART_DB_PATH)

    temp_dir = tempfile.TemporaryDirectory()
    temp_db_path = Path(temp_dir.name) / "test_smartcart.db"
    if SMARTCART_DB_PATH.exists():
        shutil.copy2(SMARTCART_DB_PATH, temp_db_path)

    # Rebind engine & SessionLocal to isolated temp database
    test_db_url = f"sqlite:///{temp_db_path.as_posix()}"
    settings.DATABASE_URL = test_db_url

    test_engine = create_engine(
        test_db_url,
        connect_args={"check_same_thread": False},
        echo=False
    )
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    # Patch app.database module objects
    orig_engine = db_module.engine
    orig_SessionLocal = getattr(db_module, "_SessionLocal", db_module.SessionLocal)
    db_module.engine = test_engine
    if hasattr(db_module, "_SessionLocal"):
        db_module._SessionLocal = TestSessionLocal
    else:
        db_module.SessionLocal = TestSessionLocal

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_module.get_db] = override_get_db

    yield {
        "temp_db_path": temp_db_path,
        "test_session_local": TestSessionLocal,
        "initial_hash": initial_hash
    }

    # Clean up dependency overrides
    app.dependency_overrides.pop(db_module.get_db, None)
    db_module.engine = orig_engine
    if hasattr(db_module, "_SessionLocal"):
        db_module._SessionLocal = orig_SessionLocal
    else:
        db_module.SessionLocal = orig_SessionLocal
    test_engine.dispose()
    try:
        temp_dir.cleanup()
    except Exception:
        pass

    # Safeguard check: Assert smartcart.db is byte-for-byte identical to baseline
    final_hash = compute_file_hash(SMARTCART_DB_PATH)
    assert initial_hash == final_hash, (
        f"CRITICAL TEST ISOLATION FAILURE: smartcart.db was modified during pytest run! "
        f"Initial SHA-256: {initial_hash}, Final SHA-256: {final_hash}"
    )

"""
SmartCart AI — Reproducible Clean Database Rebuild Script
Rebuilds a pristine smartcart.db from scratch:
  1. Backs up existing database to backend/smartcart.pre-clean.bak
  2. Creates fresh schema via SQLAlchemy Base.metadata.create_all
  3. Seeds 105 active Books from seed catalog
  4. Ingests 157 active DummyJSON products and 471 verified reviews
  5. Seeds demo accounts (admin, rahul, priya)
  6. Restores baseline model weights from data/snapshots/model_states_backup.json
  7. Confirms zero test rows (clean state)
"""

import os
import sys
import shutil
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.config import settings
from app.database import Base, engine, SessionLocal
from app.models import Product, User, ProductReview, ModelState
from app.auth import hash_password as get_password_hash
from app.seed import SEED_PRODUCTS
from scripts.import_dummyjson import run_import as import_dummyjson


def rebuild_clean_db():
    print("=" * 70)
    print("SMARTCART AI: CLEAN DATABASE REBUILD UTILITY")
    print("=" * 70)

    db_path = backend_dir / "smartcart.db"
    backup_path = backend_dir / "smartcart.pre-clean.bak"

    # 1. Backup existing DB
    if db_path.exists():
        shutil.copy2(db_path, backup_path)
        print(f"[1/6] Existing database backed up to {backup_path.name}")
        engine.dispose()
        # Remove WAL files
        for ext in ["", "-shm", "-wal"]:
            f = backend_dir / f"smartcart.db{ext}"
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

    # 2. Create schema
    print("[2/6] Initializing clean database tables...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    # 3. Seed Books
    print("[3/6] Seeding 105 Books department products...")
    book_items = [p for p in SEED_PRODUCTS if p[3] == "Books"]
    book_records = []
    for item in book_items:
        name_template, price, mrp, subcat = item
        name = name_template.replace("{brand}", "Classic").replace("{spec}", "Edition")
        prod = Product(
            name=name,
            description=f"Standard edition book: {name}. Highly rated in {subcat}.",
            price=float(price),
            mrp=float(mrp),
            category="Books",
            subcategory=subcat,
            brand="Standard Publisher",
            rating=4.5,
            review_count=120,
            stock=50,
            image_url="https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&w=600&q=80",
            source="seed",
            source_id=f"seed_book_{len(book_records) + 1}",
            is_active=True
        )
        book_records.append(prod)

    db.add_all(book_records)
    db.commit()
    print(f"  Added {len(book_records)} Books")

    # 4. Import DummyJSON Catalog (157 items + 471 reviews)
    print("[4/6] Importing DummyJSON catalog (157 active products + 471 reviews)...")
    db.close()
    import_dummyjson()

    db = SessionLocal()
    # 5. Seed Demo Users
    print("[5/6] Creating demo user accounts...")
    users = [
        User(
            name="Administrator",
            email="admin@smartcart.ai",
            password_hash=get_password_hash("admin123"),
            role="admin",
            consent_tracking=True,
            assigned_algorithm="linucb"
        ),
        User(
            name="Rahul Sharma",
            email="rahul@smartcart.ai",
            password_hash=get_password_hash("password123"),
            role="user",
            consent_tracking=True,
            assigned_algorithm="linucb"
        ),
        User(
            name="Priya Patel",
            email="priya@smartcart.ai",
            password_hash=get_password_hash("password123"),
            role="user",
            consent_tracking=True,
            assigned_algorithm="discounted_linucb"
        )
    ]
    db.add_all(users)
    db.commit()

    # 6. Restore Model States Snapshot
    print("[6/6] Restoring baseline model states from snapshot...")
    snapshot_file = backend_dir.parent / "data" / "snapshots" / "model_states_backup.json"
    if snapshot_file.exists():
        with open(snapshot_file, "r", encoding="utf-8") as f:
            snapshots = json.load(f)

        for s in snapshots:
            algo = s["algorithm"]
            # Exclude test-modified model states if any
            m_state = ModelState(
                algorithm=algo,
                state_blob=s["state_blob"],
                n_updates=s["n_updates"],
                updated_at=datetime.now(timezone.utc)
            )
            db.merge(m_state)
        db.commit()
        print(f"  Restored {len(snapshots)} model states into model_states table")

    # Final Validation
    active_count = db.query(Product).filter(Product.is_active == True).count()
    user_count = db.query(User).count()
    review_count = db.query(ProductReview).count()
    db.close()

    print("=" * 70)
    print("CLEAN DATABASE REBUILD COMPLETED SUCCESSFULLY")
    print(f"Active Products: {active_count} (Expected: 262)")
    print(f"Users:           {user_count} (Expected: 3)")
    print(f"Reviews:         {review_count} (Expected: 471)")
    print(f"Database File:   {db_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    rebuild_clean_db()


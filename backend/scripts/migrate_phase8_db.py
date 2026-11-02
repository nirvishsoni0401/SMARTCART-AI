"""
SmartCart AI — Phase 8 Database Migration Script
Adds Phase 8 data hygiene columns, origin tracking, and product_reviews table.
Migrates existing interaction_events and recommendation_logs to origin='legacy'.
"""
import sqlite3
from pathlib import Path

def run_migration(db_path: Path = None):
    if db_path is None:
        db_path = Path(__file__).resolve().parent.parent / "smartcart.db"
    
    if not db_path.exists():
        print(f"Database file not found at {db_path}. Creating fresh.")
        return

    print(f"Connecting to database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    def add_col_if_missing(table: str, col_name: str, col_def: str):
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cursor.fetchall()]
        if col_name not in cols:
            print(f"Adding column '{col_name}' to table '{table}'...")
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        else:
            print(f"Column '{col_name}' already exists in '{table}'.")

    # 1. Products table extensions
    add_col_if_missing("products", "source", "VARCHAR(50) DEFAULT 'seed' NOT NULL")
    add_col_if_missing("products", "source_id", "VARCHAR(100)")
    add_col_if_missing("products", "is_active", "BOOLEAN DEFAULT 1 NOT NULL")
    add_col_if_missing("products", "raw_price_usd", "FLOAT")
    add_col_if_missing("products", "imported_at", "DATETIME")

    # 2. Recommendation Logs extensions
    add_col_if_missing("recommendation_logs", "origin", "VARCHAR(20) DEFAULT 'live' NOT NULL")
    add_col_if_missing("recommendation_logs", "recommendation_id", "VARCHAR(50)")
    add_col_if_missing("recommendation_logs", "model_version", "VARCHAR(50) DEFAULT 'v2.0' NOT NULL")
    add_col_if_missing("recommendation_logs", "ranked_list", "JSON")
    add_col_if_missing("recommendation_logs", "feature_vector_snapshot", "JSON")

    # 3. Interaction Events extensions
    add_col_if_missing("interaction_events", "origin", "VARCHAR(20) DEFAULT 'live' NOT NULL")

    # 4. Product Reviews table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL,
        reviewer_name VARCHAR(100) NOT NULL,
        rating FLOAT NOT NULL,
        comment TEXT NOT NULL,
        date DATETIME,
        source VARCHAR(50) DEFAULT 'dummyjson' NOT NULL,
        created_at DATETIME NOT NULL,
        FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS ix_product_reviews_product_id ON product_reviews (product_id)")

    # 5. Migrate pre-existing interaction events as 'legacy'
    cursor.execute("UPDATE interaction_events SET origin = 'legacy' WHERE origin = 'live' OR origin IS NULL")
    events_count = cursor.rowcount
    print(f"Migrated {events_count} interaction events to origin='legacy'.")

    # 6. Migrate pre-existing recommendation logs as 'legacy'
    cursor.execute("UPDATE recommendation_logs SET origin = 'legacy' WHERE origin = 'live' OR origin IS NULL")
    logs_count = cursor.rowcount
    print(f"Migrated {logs_count} recommendation logs to origin='legacy'.")

    conn.commit()
    conn.close()
    print("Database migration completed successfully.")

if __name__ == "__main__":
    run_migration()

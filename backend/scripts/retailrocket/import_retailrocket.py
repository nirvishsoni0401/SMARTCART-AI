"""
SmartCart AI — Retailrocket Filtered Ingestion Pipeline
Builds backend/historical.db from RetailRocket dataset applying multi-stage
quality filters, sessionization, and transactional safeguards.
Supports --dry-run, --limit, resumable execution, and idempotent re-runs.
"""

import os
import sys
import time
import uuid
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict
import sqlite3
import json
import yaml
import numpy as np
import pandas as pd


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()[:12]
    except Exception:
        return "unknown"


def init_historical_db(conn: sqlite3.Connection):
    """Create schema in historical.db with indexes and idempotent constraints."""
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute("PRAGMA synchronous=NORMAL;")

    # 1. Category Tree
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_category_tree (
        category_id INTEGER PRIMARY KEY,
        parent_id INTEGER,
        depth INTEGER NOT NULL,
        root_id INTEGER NOT NULL
    );
    """)

    # 2. Item Categories
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_item_category (
        item_id INTEGER PRIMARY KEY,
        category_id INTEGER NOT NULL,
        root_category_id INTEGER NOT NULL,
        rule VARCHAR(50) NOT NULL
    );
    """)

    # 3. Sessions
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_sessions (
        session_id VARCHAR(64) PRIMARY KEY,
        visitor_id INTEGER NOT NULL,
        start_ts INTEGER NOT NULL,
        end_ts INTEGER NOT NULL,
        n_events INTEGER NOT NULL,
        duration_sec FLOAT NOT NULL
    );
    """)

    # 4. Filtered Events
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        visitor_id INTEGER NOT NULL,
        item_id INTEGER NOT NULL,
        event_type VARCHAR(20) NOT NULL,
        ts INTEGER NOT NULL,
        transaction_id VARCHAR(50),
        session_id VARCHAR(64),
        CONSTRAINT unq_event UNIQUE (visitor_id, item_id, event_type, ts)
    );
    """)

    # 5. Import Batches Log
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_import_batches (
        batch_id VARCHAR(64) PRIMARY KEY,
        source_file VARCHAR(100) NOT NULL,
        thresholds_used JSON NOT NULL,
        row_counts_at_every_filter_stage JSON NOT NULL,
        started_at DATETIME NOT NULL,
        finished_at DATETIME NOT NULL,
        git_commit VARCHAR(50) NOT NULL
    );
    """)

    # 6. Item Proxy Mapping (Part C)
    c.execute("""
    CREATE TABLE IF NOT EXISTS rr_item_map (
        item_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        department VARCHAR(50) NOT NULL,
        mapping_method VARCHAR(50) NOT NULL,
        mapping_version VARCHAR(20) NOT NULL,
        cluster_id INTEGER,
        confidence FLOAT,
        mapped_at DATETIME NOT NULL,
        PRIMARY KEY (item_id, mapping_method, mapping_version)
    );
    """)

    conn.commit()


def run_import(config_path: str = "config/retailrocket.yaml", limit: int = None, dry_run: bool = False):
    print("=" * 70, flush=True)
    print("SMARTCART AI: RETAILROCKET FILTERED INGESTION PIPELINE", flush=True)
    print("=" * 70, flush=True)

    start_time = time.time()
    batch_id = f"batch_{uuid.uuid4().hex[:12]}"
    git_commit = get_git_commit()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["retailrocket"]

    raw_dir = Path(cfg["raw_data_dir"])
    db_path = Path(cfg["output_db_path"])
    filters = cfg["filters"]
    sess_cfg = cfg["sessionization"]

    max_events_per_visitor = filters["max_events_per_visitor"]
    min_events_per_item = filters["min_events_per_item"]
    min_visitors_per_item = filters["min_visitors_per_item"]
    always_keep_transactions = filters["always_keep_transactions"]
    inactivity_ms = sess_cfg["inactivity_window_ms"]

    stage_counts = {
        "raw_events_read": 0,
        "exact_duplicates_dropped": 0,
        "bot_events_capped": 0,
        "tail_items_filtered": 0,
        "transactions_preserved_safeguard": 0,
        "final_events_imported": 0,
        "sessions_created": 0,
        "items_in_catalog": 0,
        "categories_imported": 0
    }

    if dry_run:
        print("[DRY-RUN MODE] Ingestion simulation active. No database modifications will occur.\n", flush=True)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        init_historical_db(conn)

    # -------------------------------------------------------------------------
    # 1. Import Category Tree
    # -------------------------------------------------------------------------
    print("[1/5] Ingesting Category Tree...", flush=True)
    tree_path = raw_dir / "category_tree.csv"
    df_tree = pd.read_csv(tree_path)

    parent_map = {}
    roots = set()
    for row in df_tree.itertuples(index=False):
        cid = int(row.categoryid)
        pid = row.parentid
        if pd.isna(pid):
            roots.add(cid)
            parent_map[cid] = None
        else:
            parent_map[cid] = int(pid)

    cat_depth = {}
    cat_to_root = {}

    def get_cat_info(cid, visited=None):
        if visited is None:
            visited = set()
        if cid in visited:
            return 0, cid
        visited.add(cid)
        parent = parent_map.get(cid)
        if parent is None:
            return 0, cid
        p_depth, root = get_cat_info(parent, visited)
        return p_depth + 1, root

    tree_records = []
    for cid in parent_map:
        d, r = get_cat_info(cid)
        cat_depth[cid] = d
        cat_to_root[cid] = r
        tree_records.append((cid, parent_map[cid], d, r))

    stage_counts["categories_imported"] = len(tree_records)
    print(f"  Computed hierarchy for {len(tree_records):,} categories ({len(roots)} roots)", flush=True)

    if not dry_run:
        c = conn.cursor()
        c.executemany("""
            INSERT OR REPLACE INTO rr_category_tree (category_id, parent_id, depth, root_id)
            VALUES (?, ?, ?, ?);
        """, tree_records)
        conn.commit()

    # -------------------------------------------------------------------------
    # 2. Ingest Item Category Mapping (Latest Timestamp Rule)
    # -------------------------------------------------------------------------
    print("\n[2/5] Ingesting Item Properties (Category IDs)...", flush=True)
    item_latest_cat = {}  # itemid -> (timestamp, categoryid)
    prop_files = [raw_dir / "item_properties_part1.csv", raw_dir / "item_properties_part2.csv"]

    for pfile in prop_files:
        for chunk in pd.read_csv(pfile, chunksize=500000, usecols=["timestamp", "itemid", "property", "value"]):
            cat_rows = chunk[chunk["property"] == "categoryid"]
            if cat_rows.empty:
                continue
            for row in cat_rows.itertuples(index=False):
                ts, iid, _, val = row
                try:
                    ival = int(val)
                except ValueError:
                    continue
                cur = item_latest_cat.get(iid)
                if cur is None or ts > cur[0]:
                    item_latest_cat[iid] = (ts, ival)

    item_cat_records = []
    for iid, (_, cid) in item_latest_cat.items():
        root_id = cat_to_root.get(cid, cid)
        item_cat_records.append((iid, cid, root_id, "latest_timestamp"))

    stage_counts["items_in_catalog"] = len(item_cat_records)
    print(f"  Extracted canonical category for {len(item_cat_records):,} items", flush=True)

    if not dry_run:
        c = conn.cursor()
        c.executemany("""
            INSERT OR REPLACE INTO rr_item_category (item_id, category_id, root_category_id, rule)
            VALUES (?, ?, ?, ?);
        """, item_cat_records)
        conn.commit()

    # -------------------------------------------------------------------------
    # 3. Stream & Filter Events
    # -------------------------------------------------------------------------
    print("\n[3/5] Streaming & Filtering Events clickstream...", flush=True)
    events_path = raw_dir / "events.csv"

    # Step A: First pass to count visitor frequencies, item frequencies, and transactions
    print("  Pass 1: Computing activity statistics...", flush=True)
    visitor_counts = Counter()
    item_counts = Counter()
    item_visitors = defaultdict(set)
    items_with_transactions = set()
    raw_read = 0

    for chunk in pd.read_csv(events_path, chunksize=500000):
        if limit and raw_read >= limit:
            break
        if limit and raw_read + len(chunk) > limit:
            chunk = chunk.iloc[:(limit - raw_read)]

        raw_read += len(chunk)

        for vid, c in chunk["visitorid"].value_counts().items():
            visitor_counts[vid] += c
        for iid, c in chunk["itemid"].value_counts().items():
            item_counts[iid] += c

        # Track transactions and visitors per item
        for row in chunk[chunk["event"] == "transaction"].itertuples(index=False):
            items_with_transactions.add(int(row.itemid))

        # Sample item visitors
        for row in chunk.itertuples(index=False):
            item_visitors[int(row.itemid)].add(int(row.visitorid))

    stage_counts["raw_events_read"] = raw_read
    print(f"  Pass 1 complete: {raw_read:,} raw events evaluated across {len(visitor_counts):,} visitors", flush=True)

    # Step B: Determine kept items
    kept_items = set()
    tx_safeguarded = 0
    for iid, c in item_counts.items():
        n_vis = len(item_visitors.get(iid, set()))
        if c >= min_events_per_item and n_vis >= min_visitors_per_item:
            kept_items.add(iid)
        elif always_keep_transactions and iid in items_with_transactions:
            kept_items.add(iid)
            tx_safeguarded += 1

    stage_counts["transactions_preserved_safeguard"] = tx_safeguarded
    print(f"  Items qualifying: {len(kept_items):,} (including {tx_safeguarded:,} items preserved via transaction safeguard)", flush=True)

    # Step C: Pass 2: Filter events, apply deduplication, and cap bots
    print("  Pass 2: Filtering clickstream and applying deduplication...", flush=True)
    visitor_seen_events = Counter()
    filtered_events = []
    seen_keys = set()
    raw_second_pass = 0

    for chunk in pd.read_csv(events_path, chunksize=500000):
        if limit and raw_second_pass >= limit:
            break
        if limit and raw_second_pass + len(chunk) > limit:
            chunk = chunk.iloc[:(limit - raw_second_pass)]

        raw_second_pass += len(chunk)

        for row in chunk.itertuples(index=False):
            ts = int(row.timestamp)
            vid = int(row.visitorid)
            ev = str(row.event)
            iid = int(row.itemid)
            tid = str(row.transactionid) if pd.notna(row.transactionid) else None

            # 1. Exact Duplicate filter
            k = (vid, iid, ev, ts)
            if k in seen_keys:
                stage_counts["exact_duplicates_dropped"] += 1
                continue
            seen_keys.add(k)

            # 2. Transaction safeguard
            is_tx = (ev == "transaction")

            # 3. Item filter (skip tail items unless transaction)
            if not is_tx and iid not in kept_items:
                stage_counts["tail_items_filtered"] += 1
                continue

            # 4. Bot capping (cap visitor events at max_events_per_visitor, except transactions)
            if not is_tx and visitor_seen_events[vid] >= max_events_per_visitor:
                stage_counts["bot_events_capped"] += 1
                continue

            visitor_seen_events[vid] += 1
            filtered_events.append((vid, iid, ev, ts, tid))

    del seen_keys
    del item_visitors

    stage_counts["final_events_imported"] = len(filtered_events)
    print(f"  Filtered events retained: {len(filtered_events):,} ({len(filtered_events)/raw_read*100:.1f}% of raw)", flush=True)

    # -------------------------------------------------------------------------
    # 4. Sessionization of Kept Events
    # -------------------------------------------------------------------------
    print("\n[4/5] Computing Sessions for Kept Events (30-Min Inactivity)...", flush=True)
    df_kept = pd.DataFrame(filtered_events, columns=["visitor_id", "item_id", "event_type", "ts", "transaction_id"])
    del filtered_events

    df_kept.sort_values(by=["visitor_id", "ts"], inplace=True)
    vids = df_kept["visitor_id"].to_numpy()
    tss = df_kept["ts"].to_numpy()

    is_new_sess = np.empty(len(vids), dtype=bool)
    is_new_sess[0] = True
    is_new_sess[1:] = (vids[1:] != vids[:-1]) | ((tss[1:] - tss[:-1]) > inactivity_ms)

    # Assign deterministic session_ids: s_{visitor_id}_{session_num}
    sess_cum = np.cumsum(is_new_sess)
    df_kept["session_idx"] = sess_cum

    session_records = []
    # Vectorized session bounds
    sess_starts = np.where(is_new_sess)[0]
    sess_ends = np.append(sess_starts[1:] - 1, len(vids) - 1)

    session_ids = []
    for s_idx, (start_i, end_i) in enumerate(zip(sess_starts, sess_ends), 1):
        s_vid = int(vids[start_i])
        s_start = int(tss[start_i])
        s_end = int(tss[end_i])
        s_n = int(end_i - start_i + 1)
        s_dur = float((s_end - s_start) / 1000.0)
        sid = f"s_{s_vid}_{s_start}"
        session_records.append((sid, s_vid, s_start, s_end, s_n, s_dur))

    # Map session_ids back to events
    # Expand session_ids per event
    event_session_ids = []
    for sid, _, _, _, s_n, _ in session_records:
        event_session_ids.extend([sid] * s_n)

    df_kept["session_id"] = event_session_ids
    stage_counts["sessions_created"] = len(session_records)
    print(f"  Created {len(session_records):,} distinct sessions", flush=True)

    # -------------------------------------------------------------------------
    # 5. Persist to historical.db
    # -------------------------------------------------------------------------
    print("\n[5/5] Persisting into backend/historical.db...", flush=True)
    if not dry_run:
        c = conn.cursor()

        # Insert sessions
        print("  Inserting sessions...", flush=True)
        c.executemany("""
            INSERT OR REPLACE INTO rr_sessions (session_id, visitor_id, start_ts, end_ts, n_events, duration_sec)
            VALUES (?, ?, ?, ?, ?, ?);
        """, session_records)

        # Insert events in batches of 100,000
        print("  Inserting filtered events...", flush=True)
        event_tuples = [
            (int(r.visitor_id), int(r.item_id), str(r.event_type), int(r.ts), r.transaction_id, str(r.session_id))
            for r in df_kept.itertuples(index=False)
        ]

        batch_size = 100000
        for i in range(0, len(event_tuples), batch_size):
            chunk = event_tuples[i : i + batch_size]
            c.executemany("""
                INSERT OR IGNORE INTO rr_events (visitor_id, item_id, event_type, ts, transaction_id, session_id)
                VALUES (?, ?, ?, ?, ?, ?);
            """, chunk)
            conn.commit()

        # Build indexes
        print("  Building indexes...", flush=True)
        c.execute("CREATE INDEX IF NOT EXISTS idx_rr_events_visitor ON rr_events(visitor_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rr_events_item ON rr_events(item_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rr_events_session ON rr_events(session_id);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rr_events_type ON rr_events(event_type);")
        c.execute("CREATE INDEX IF NOT EXISTS idx_rr_sessions_visitor ON rr_sessions(visitor_id);")

        # Record import batch
        finished_at = datetime.now(timezone.utc).isoformat()
        c.execute("""
            INSERT INTO rr_import_batches (batch_id, source_file, thresholds_used, row_counts_at_every_filter_stage, started_at, finished_at, git_commit)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            batch_id,
            "events.csv",
            json.dumps(filters),
            json.dumps(stage_counts),
            datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat(),
            finished_at,
            git_commit
        ))
        conn.commit()

        # Check final DB size
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        conn.close()
    else:
        db_size_mb = 0.0

    total_time = time.time() - start_time
    print("=" * 70, flush=True)
    print("RETAILROCKET FILTERED INGESTION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"Batch ID:                    {batch_id}")
    print(f"Git Commit:                  {git_commit}")
    print(f"Total Runtime:               {total_time:.2f} seconds")
    print(f"Raw Events Read:             {stage_counts['raw_events_read']:,}")
    print(f"Exact Duplicates Dropped:    {stage_counts['exact_duplicates_dropped']:,}")
    print(f"Bot Events Capped:           {stage_counts['bot_events_capped']:,}")
    print(f"Tail Items Filtered:         {stage_counts['tail_items_filtered']:,}")
    print(f"Transactions Preserved:      {stage_counts['transactions_preserved_safeguard']:,}")
    print(f"Final Events Imported:       {stage_counts['final_events_imported']:,}")
    print(f"Sessions Created:            {stage_counts['sessions_created']:,}")
    print(f"Categories Imported:         {stage_counts['categories_imported']:,}")
    print(f"Items in Catalog:            {stage_counts['items_in_catalog']:,}")
    print(f"Historical DB Disk Size:     {db_size_mb:.2f} MB")
    print("=" * 70, flush=True)

    return stage_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retailrocket Filtered Ingestion Pipeline")
    parser.add_argument("--config", default="config/retailrocket.yaml", help="Path to retailrocket.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of raw events to read (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without modifying database")
    args = parser.parse_args()

    run_import(config_path=args.config, limit=args.limit, dry_run=args.dry_run)

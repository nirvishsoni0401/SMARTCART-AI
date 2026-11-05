import sys
import sqlite3
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.feature_engineering import (
    FEATURE_DIMENSION,
    FEATURE_SPEC_HASH,
    CATEGORIES,
    CATEGORY_INDEX,
    UserContext,
    build_context_from_history,
    extract_feature_vector
)

HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
T_VAL_END = 1440756710277

def audit_session_features():
    print("=== AUDIT OF SESSION FEATURES ===")
    conn_sc = sqlite3.connect(SMARTCART_DB)
    c_sc = conn_sc.cursor()
    prods = {}
    for r in c_sc.execute("SELECT id, name, category, brand, price, rating, discount_pct, popularity_score FROM products WHERE is_active=1").fetchall():
        prods[r[0]] = {
            "id": r[0], "name": r[1], "category": r[2], "brand": r[3] or "",
            "price": float(r[4]), "rating": float(r[5]), "discount_pct": float(r[6]),
            "popularity_score": float(r[7])
        }
    conn_sc.close()

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    mapping = dict(c_hist.execute("""
        SELECT item_id, product_id FROM rr_item_map 
        WHERE mapping_method='popularity_balanced_train' AND mapping_version='v1.0_train'
    """).fetchall())

    # Let's inspect test events for a sample visitor who has view -> addtocart -> transaction
    rows = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts 
        FROM rr_events 
        WHERE ts >= ?
        ORDER BY visitor_id, ts
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    # Find visitors with multi-event sessions including addtocart and transaction
    vis_events = defaultdict(list)
    for v_id, i_id, ev_type, ts in rows:
        pid = mapping.get(i_id)
        if pid in prods:
            vis_events[v_id].append((pid, i_id, ev_type, ts))

    print(f"Total unique test visitors with valid events: {len(vis_events):,}")
    
    # Check visitors who have transaction
    trans_visitors = [v for v, evs in vis_events.items() if any(e[2] == 'transaction' for e in evs)]
    print(f"Visitors with transaction in test: {len(trans_visitors):,}")

    # Inspect a few transaction sessions
    sample_trans_cases = []
    for v in trans_visitors:
        evs = vis_events[v]
        # check if transaction was preceded by addtocart
        types = [e[2] for e in evs]
        if 'addtocart' in types and 'transaction' in types:
            sample_trans_cases.append((v, evs))
            if len(sample_trans_cases) >= 5:
                break

    for v, evs in sample_trans_cases:
        print(f"\nVisitor {v} event sequence:")
        for pid, i_id, ev_type, ts in evs:
            p = prods[pid]
            print(f"  ts={ts}: event={ev_type:12s} item_id={i_id} -> pid={pid} ({p['name'][:30]}, cat={p['category']}, brand={p['brand']})")

if __name__ == "__main__":
    audit_session_features()


"""
SmartCart AI — Populate Catalog Popularity Prior from Historical Train Interactions
"""
import sqlite3
from collections import Counter
from pathlib import Path

HISTORICAL_DB = Path("backend/historical.db")
SMARTCART_DB = Path("backend/smartcart.db")
TRAIN_CUTOFF_MS = 1438968232766

def populate_prior():
    conn_h = sqlite3.connect(HISTORICAL_DB)
    c_h = conn_h.cursor()
    conn_s = sqlite3.connect(SMARTCART_DB)
    c_s = conn_s.cursor()

    item_map = dict(c_h.execute("""
        SELECT item_id, product_id 
        FROM rr_item_map 
        WHERE mapping_method = 'popularity_balanced_train' AND mapping_version = 'v1.0_train';
    """).fetchall())

    evs = c_h.execute("SELECT item_id, COUNT(*) FROM rr_events WHERE ts < ? GROUP BY item_id;", (TRAIN_CUTOFF_MS,)).fetchall()

    prod_counts = Counter()
    for iid, cnt in evs:
        pid = item_map.get(iid)
        if pid is not None:
            prod_counts[pid] += cnt

    all_pids = [r[0] for r in c_s.execute("SELECT id FROM products WHERE is_active=1;").fetchall()]
    max_c = max(prod_counts.values()) if prod_counts else 1.0

    for pid in all_pids:
        cnt = prod_counts.get(pid, 0)
        score = round((cnt / max_c) * 100.0, 2)
        c_s.execute("UPDATE products SET popularity_prior = ? WHERE id = ?;", (score, pid))

    conn_s.commit()
    conn_h.close()
    conn_s.close()
    print(f"Popularity priors populated for {len(all_pids)} active products. Max events count was {max_c:,}.")

if __name__ == "__main__":
    populate_prior()


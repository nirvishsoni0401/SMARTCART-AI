import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
HISTORICAL_DB = BACKEND_DIR / "historical.db"
TRAIN_CUTOFF_MS = 1438968232766

def check_train_mappings():
    conn = sqlite3.connect(HISTORICAL_DB)
    c = conn.cursor()

    # 1. Check date range of rr_events where ts < TRAIN_CUTOFF_MS
    min_ts, max_ts, cnt = c.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM rr_events WHERE ts < ?;", (TRAIN_CUTOFF_MS,)).fetchone()
    dt_min = datetime.fromtimestamp(min_ts/1000.0, tz=timezone.utc)
    dt_max = datetime.fromtimestamp(max_ts/1000.0, tz=timezone.utc)
    dt_cutoff = datetime.fromtimestamp(TRAIN_CUTOFF_MS/1000.0, tz=timezone.utc)
    print(f"Train Events Range:")
    print(f"  Min TS: {min_ts} ({dt_min.isoformat()})")
    print(f"  Max TS: {max_ts} ({dt_max.isoformat()})")
    print(f"  Cutoff: {TRAIN_CUTOFF_MS} ({dt_cutoff.isoformat()})")
    print(f"  Count: {cnt:,}")

    # 2. Check date range of rr_sessions where start_ts < TRAIN_CUTOFF_MS
    min_sess, max_sess, cnt_sess = c.execute("SELECT MIN(start_ts), MAX(start_ts), COUNT(*) FROM rr_sessions WHERE start_ts < ?;", (TRAIN_CUTOFF_MS,)).fetchone()
    print(f"\nTrain Sessions Range:")
    print(f"  Min Start: {min_sess} ({datetime.fromtimestamp(min_sess/1000.0, tz=timezone.utc).isoformat()})")
    print(f"  Max Start: {max_sess} ({datetime.fromtimestamp(max_sess/1000.0, tz=timezone.utc).isoformat()})")
    print(f"  Count: {cnt_sess:,}")

    # 3. Check actual Gini and top-10 share on TRAIN events strictly vs FULL events for all 3 train mappings
    train_events = dict(c.execute("SELECT item_id, COUNT(*) FROM rr_events WHERE ts < ? GROUP BY item_id;", (TRAIN_CUTOFF_MS,)).fetchall())
    total_train_ev = sum(train_events.values())

    from collections import Counter
    import numpy as np

    def gini(arr):
        arr = np.sort(np.array(arr, dtype=np.float64))
        n = len(arr)
        idx = np.arange(1, n + 1)
        return float((np.sum((2 * idx - n - 1) * arr)) / (n * np.sum(arr)))

    methods = [
        ("category_rank_train", "v1.0_train"),
        ("cooccurrence_cluster_train", "v1.0_train"),
        ("popularity_balanced_train", "v1.0_train")
    ]

    print("\nStrict In-Train Gini & Top-10 Concentration for Train Mappings:")
    for method, ver in methods:
        m_dict = dict(c.execute("SELECT item_id, product_id FROM rr_item_map WHERE mapping_method=? AND mapping_version=?;", (method, ver)).fetchall())
        prod_counts = Counter()
        for iid, pid in m_dict.items():
            prod_counts[pid] += train_events.get(iid, 0)
        arr = np.array(list(prod_counts.values()))
        arr_sorted = np.sort(arr)[::-1]
        g = gini(arr)
        top1 = arr_sorted[0] / total_train_ev * 100.0
        top10 = np.sum(arr_sorted[:10]) / total_train_ev * 100.0
        print(f"  {method:28s}: In-Train Gini={g:.4f}, Top-1={top1:.2f}%, Top-10={top10:.2f}% (active SKUs={len(prod_counts)})")

    conn.close()

if __name__ == "__main__":
    check_train_mappings()


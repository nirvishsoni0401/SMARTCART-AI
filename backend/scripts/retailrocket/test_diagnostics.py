import sqlite3
import numpy as np
from collections import Counter, defaultdict

def gini(array):
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 1e-7
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

conn_h = sqlite3.connect('backend/historical.db')
c_h = conn_h.cursor()
conn_s = sqlite3.connect('backend/smartcart.db')
c_s = conn_s.cursor()

item_events = dict(c_h.execute('SELECT item_id, COUNT(*) FROM rr_events GROUP BY item_id').fetchall())

dept_prods = defaultdict(list)
for pid, cat in c_s.execute('SELECT id, category FROM products WHERE is_active=1').fetchall():
    dept_prods[cat].append(pid)

print("Catalog SKU share by department:")
total_skus = sum(len(p) for p in dept_prods.values())
for dept, p in sorted(dept_prods.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {dept:16s}: {len(p)} SKUs ({len(p)/total_skus*100:.1f}%)")

for method in ['category_rank', 'cooccurrence_cluster']:
    rows = c_h.execute('SELECT item_id, product_id, department, cluster_id, confidence FROM rr_item_map WHERE mapping_method = ? AND mapping_version = "v1.0"', (method,)).fetchall()
    prod_events = Counter()
    dept_events = Counter()
    cluster_sizes = Counter()
    no_edge_count = 0
    
    for iid, pid, dept, cid, conf in rows:
        evs = item_events.get(iid, 0)
        prod_events[pid] += evs
        dept_events[dept] += evs
        cluster_sizes[cid] += 1
        if conf == 0.60: # fallback confidence
            no_edge_count += 1
            
    counts = np.array(list(prod_events.values()))
    total = np.sum(counts)
    sorted_counts = np.sort(counts)[::-1]
    
    print(f"\n--- Method: {method} ---")
    print(f"  Gini: {gini(counts):.4f}")
    print(f"  Top-1 share: {sorted_counts[0] / total * 100:.2f}%")
    print(f"  Top-10 share: {np.sum(sorted_counts[:10]) / total * 100:.2f}%")
    print(f"  Items with no co-occurrence embedding: {no_edge_count:,} ({no_edge_count/len(rows)*100:.2f}%)")
    print("  Cluster sizes:", dict(cluster_sizes.most_common(6)))
    print("  Event share by department:")
    for dept, evs in sorted(dept_events.items(), key=lambda x: x[1], reverse=True):
        print(f"    {dept:16s}: {evs:,} ({evs/total*100:.2f}%)")

# Test popularity_balanced assignment
rows_cc = c_h.execute('SELECT item_id, department, cluster_id FROM rr_item_map WHERE mapping_method = "cooccurrence_cluster" AND mapping_version = "v1.0"').fetchall()
dept_items = defaultdict(list)
for iid, dept, cid in rows_cc:
    dept_items[dept].append(iid)

balanced_prod_events = Counter()
for dept, items in dept_items.items():
    prods = dept_prods[dept]
    n_p = len(prods)
    s_items = sorted(items, key=lambda x: item_events.get(x, 0), reverse=True)
    for rank, iid in enumerate(s_items):
        cycle = rank // n_p
        rem = rank % n_p
        p_idx = rem if cycle % 2 == 0 else (n_p - 1 - rem)
        pid = prods[p_idx]
        balanced_prod_events[pid] += item_events.get(iid, 0)

b_counts = np.array(list(balanced_prod_events.values()))
b_total = np.sum(b_counts)
b_sorted = np.sort(b_counts)[::-1]

print(f"\n--- Method: popularity_balanced (Snake Stride) ---")
print(f"  Gini: {gini(b_counts):.4f}")
print(f"  Top-1 share: {b_sorted[0] / b_total * 100:.2f}%")
print(f"  Top-10 share: {np.sum(b_sorted[:10]) / b_total * 100:.2f}%")

conn_h.close()
conn_s.close()


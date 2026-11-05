import sys
import sqlite3
from collections import defaultdict, Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
HISTORICAL_DB = BACKEND_DIR / "historical.db"
SMARTCART_DB = BACKEND_DIR / "smartcart.db"
T_VAL_END = 1440756710277

def main():
    conn_sc = sqlite3.connect(SMARTCART_DB)
    c_sc = conn_sc.cursor()
    catalog_pids = set(r[0] for r in c_sc.execute("SELECT id FROM products WHERE is_active=1").fetchall())
    conn_sc.close()

    conn_hist = sqlite3.connect(HISTORICAL_DB)
    c_hist = conn_hist.cursor()
    mapping = dict(c_hist.execute("""
        SELECT item_id, product_id FROM rr_item_map 
        WHERE mapping_method='popularity_balanced_train' AND mapping_version='v1.0_train'
    """).fetchall())

    test_events = c_hist.execute("""
        SELECT visitor_id, item_id, event_type, ts 
        FROM rr_events 
        WHERE ts >= ? 
        ORDER BY visitor_id, ts
    """, (T_VAL_END,)).fetchall()
    conn_hist.close()

    print(f"Total test events in DB: {len(test_events):,}")
    valid_events = [e for e in test_events if mapping.get(e[1]) in catalog_pids]
    print(f"Valid test events (mapped to active catalog): {len(valid_events):,}")

    n_repeat_prod = 0
    n_repeat_same_item = 0
    n_repeat_aliased = 0
    n_in_cart_same_session = 0
    n_purchased_same_session = 0
    n_in_cart_any_time = 0

    visitor_prods_all = defaultdict(set)
    visitor_items_all = defaultdict(set)

    curr_v = None
    curr_sess_ts = 0
    curr_cart_pids = set()
    curr_purchased_pids = set()
    post_cart_purchases = 0

    # Also let's check the 25k slice from Phase 9C
    slice25k_trans_add = 0
    slice25k_post_cart_purchases = 0

    for idx, (v_id, i_id, ev_type, ts) in enumerate(valid_events):
        pid = mapping[i_id]
        
        # New visitor or new session (gap > 30 min)
        if v_id != curr_v or (ts - curr_sess_ts > 1800000):
            curr_v = v_id
            curr_cart_pids = set()
            curr_purchased_pids = set()
        curr_sess_ts = ts
        
        is_repeat_prod = pid in visitor_prods_all[v_id]
        is_repeat_item = i_id in visitor_items_all[v_id]
        
        if is_repeat_prod:
            n_repeat_prod += 1
            if is_repeat_item:
                n_repeat_same_item += 1
            else:
                n_repeat_aliased += 1
                
        if pid in curr_cart_pids:
            n_in_cart_same_session += 1
        if pid in curr_purchased_pids:
            n_purchased_same_session += 1
            
        if ev_type == "transaction" and pid in curr_cart_pids:
            post_cart_purchases += 1

        if idx < 25000:
            if ev_type in ("transaction", "addtocart"):
                slice25k_trans_add += 1
            if ev_type == "transaction" and pid in curr_cart_pids:
                slice25k_post_cart_purchases += 1
            
        visitor_prods_all[v_id].add(pid)
        visitor_items_all[v_id].add(i_id)
        if ev_type in ("addtocart", "cart_add"):
            curr_cart_pids.add(pid)
        elif ev_type in ("transaction", "purchase"):
            curr_purchased_pids.add(pid)

    print(f"Repeat target products: {n_repeat_prod:,} / {len(valid_events):,} ({n_repeat_prod/len(valid_events)*100:.2f}%)")
    print(f"  - Same RR item repeat: {n_repeat_same_item:,} ({n_repeat_same_item/len(valid_events)*100:.2f}%)")
    print(f"  - Aliased item repeat: {n_repeat_aliased:,} ({n_repeat_aliased/len(valid_events)*100:.2f}%)")
    print(f"Target already in cart in same session: {n_in_cart_same_session:,}")
    print(f"Target already purchased in same session: {n_purchased_same_session:,}")
    print(f"Post-cart purchases in same session (all test): {post_cart_purchases:,}")
    print(f"First 25k slice high intent (trans + addtocart): {slice25k_trans_add:,}")
    print(f"First 25k slice post-cart purchases: {slice25k_post_cart_purchases:,}")

if __name__ == "__main__":
    main()


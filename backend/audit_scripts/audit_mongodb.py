import sys
import json
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import db, products_collection

def run_mongodb_audit():
    print("===================================================================")
    print("               1. MONGODB DATABASE AUDIT REPORT                    ")
    print("===================================================================")
    
    total_docs = products_collection.count_documents({})
    active_docs = products_collection.count_documents({"is_active": True})
    inactive_docs = products_collection.count_documents({"is_active": False})
    missing_active_flag = products_collection.count_documents({"is_active": {"$exists": False}})
    
    print(f"Total Products in DB:       {total_docs:,}")
    print(f"Active Products:            {active_docs:,}")
    print(f"Inactive Products:          {inactive_docs:,}")
    print(f"Missing 'is_active' flag:   {missing_active_flag:,}")
    
    # 1. Product count per department (category)
    pipeline_cat = [
        {"$group": {"_id": "$category", "count": {"$sum": 1}, "avg_price": {"$avg": "$price"}, "avg_discounted": {"$avg": "$discounted_price"}}},
        {"$sort": {"count": -1}}
    ]
    cat_counts = list(products_collection.aggregate(pipeline_cat))
    print(f"\n--- Product Count per Department ({len(cat_counts)} Departments) ---")
    for c in cat_counts:
        print(f"  {c['_id']:<26}: {c['count']:>5} items | MRP avg: Rs. {c['avg_price']:>9.2f} | Selling avg: Rs. {c['avg_discounted']:>9.2f}")
        
    # 2. Product count per subcategory
    pipeline_sub = [
        {"$group": {"_id": {"cat": "$category", "sub": "$subcategory"}, "count": {"$sum": 1}}},
        {"$sort": {"_id.cat": 1, "_id.sub": 1}}
    ]
    sub_counts = list(products_collection.aggregate(pipeline_sub))
    print(f"\n--- Total Subcategories Found: {len(sub_counts)} ---")
    subcat_distribution = [s["count"] for s in sub_counts]
    print(f"  Min items per subcategory: {min(subcat_distribution)}")
    print(f"  Max items per subcategory: {max(subcat_distribution)}")
    print(f"  Subcategories count per cat:")
    sub_by_cat = defaultdict(list)
    for s in sub_counts:
        sub_by_cat[s["_id"]["cat"]].append(f"{s['_id']['sub']} ({s['count']})")
    for cat, subs in sub_by_cat.items():
        print(f"    [{cat}]: {len(subs)} subcats -> {', '.join(subs[:4])}...")

    # 3. Unique brands
    brands = products_collection.distinct("brand")
    valid_brands = [b for b in brands if b and str(b).strip() and b != "Unknown"]
    print(f"\n--- Brands Audit ---")
    print(f"  Total Unique Brand strings: {len(brands)}")
    print(f"  Valid Named Brands:         {len(valid_brands)}")
    print(f"  Sample Brands:              {valid_brands[:12]}")

    # 4. Duplicate product IDs
    print(f"\n--- Duplicate IDs Check ---")
    pipeline_dups_pid = [
        {"$group": {"_id": "$product_id", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}}
    ]
    dups_pid = list(products_collection.aggregate(pipeline_dups_pid))
    print(f"  Duplicate 'product_id' count: {len(dups_pid)}")
    
    pipeline_dups_ext = [
        {"$group": {"_id": "$external_id", "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}}
    ]
    dups_ext = list(products_collection.aggregate(pipeline_dups_ext))
    print(f"  Duplicate 'external_id' count: {len(dups_ext)}")
    if dups_pid:
        print(f"    Duplicate product_ids: {dups_pid[:5]}")

    # 5. Missing required fields check
    required_fields = [
        "product_id", "title", "description", "brand", "category", "subcategory",
        "price", "discounted_price", "discount_percentage", "rating", "review_count",
        "stock_quantity", "availability", "images", "thumbnail", "seller",
        "specifications", "tags", "popularity_score", "created_at", "updated_at"
    ]
    print(f"\n--- Missing Required Fields Audit ---")
    missing_report = {}
    for f in required_fields:
        missing_cnt = products_collection.count_documents({
            "$or": [
                {f: {"$exists": False}},
                {f: None}
            ]
        })
        missing_report[f] = missing_cnt
        if missing_cnt > 0:
            print(f"  WARNING: Field '{f}' is missing in {missing_cnt} documents!")
        else:
            print(f"  Field '{f}': 0 missing (100% complete)")

    # 6. Invalid prices and discounts
    print(f"\n--- Price & Discount Integrity Check ---")
    invalid_price = products_collection.count_documents({"$or": [{"price": {"$lte": 0}}, {"price": {"$type": "string"}}]})
    invalid_disc_price = products_collection.count_documents({"$or": [{"discounted_price": {"$lte": 0}}, {"discounted_price": {"$type": "string"}}]})
    disc_higher_than_mrp = products_collection.count_documents({"$expr": {"$gt": ["$discounted_price", "$price"]}})
    invalid_disc_pct = products_collection.count_documents({"$or": [{"discount_percentage": {"$lt": 0}}, {"discount_percentage": {"$gt": 100}}]})
    
    print(f"  Invalid / Non-positive 'price':           {invalid_price}")
    print(f"  Invalid / Non-positive 'discounted_price':{invalid_disc_price}")
    print(f"  Discounted price higher than MRP:         {disc_higher_than_mrp}")
    print(f"  Invalid 'discount_percentage' (<0 or >100):{invalid_disc_pct}")

    # 7. Invalid stock values
    print(f"\n--- Stock Values Integrity Check ---")
    invalid_stock = products_collection.count_documents({"$or": [{"stock_quantity": {"$lt": 0}}, {"stock": {"$lt": 0}}]})
    mismatched_availability = products_collection.count_documents({
        "$and": [
            {"stock_quantity": {"$gt": 10}},
            {"availability": {"$ne": "In Stock"}}
        ]
    })
    print(f"  Negative stock count:                      {invalid_stock}")
    print(f"  Mismatched availability status:           {mismatched_availability}")

    # 8. Broken or missing image URLs
    print(f"\n--- Product Images Audit ---")
    empty_images = products_collection.count_documents({"$or": [{"images": {"$size": 0}}, {"images": {"$exists": False}}]})
    non_list_images = products_collection.count_documents({"images": {"$not": {"$type": "array"}}})
    empty_thumbnail = products_collection.count_documents({"$or": [{"thumbnail": ""}, {"thumbnail": {"$exists": False}}]})
    print(f"  Empty 'images' array:                      {empty_images}")
    print(f"  Non-array 'images':                        {non_list_images}")
    print(f"  Empty 'thumbnail':                         {empty_thumbnail}")

    # Test HTTP reachability of sample image URLs
    sample_prods = list(products_collection.find({}, {"thumbnail": 1, "images": 1, "title": 1}).limit(10))
    print(f"\n--- Testing HTTP Reachability of 10 Sample Product Images ---")
    img_success = 0
    img_fail = 0
    for p in sample_prods:
        url = p.get("thumbnail")
        if url:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4) as response:
                    code = response.getcode()
                    if code == 200:
                        img_success += 1
                    else:
                        img_fail += 1
            except Exception as e:
                img_fail += 1
                print(f"  Failed image URL: {url} -> {e}")
    print(f"  Image URLs reachable (HTTP 200): {img_success}/10 (Failures: {img_fail})")

if __name__ == "__main__":
    run_mongodb_audit()

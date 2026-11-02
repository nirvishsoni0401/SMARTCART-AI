import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import products_collection

# 1. Price calculations
total_docs = products_collection.count_documents({"is_active": True})
all_active = list(products_collection.find({"is_active": True}, {"price": 1, "discounted_price": 1, "category": 1, "subcategory": 1}))

sum_mrp = sum(p["price"] for p in all_active)
sum_discounted = sum(p["discounted_price"] for p in all_active)

overall_avg_mrp = sum_mrp / total_docs
overall_avg_discounted = sum_discounted / total_docs

print(f"Total Active Count: {total_docs}")
print(f"Overall Average MRP (Original Price):      Rs. {overall_avg_mrp:.2f}")
print(f"Overall Average Selling (Discounted Price): Rs. {overall_avg_discounted:.2f}")

# Weighted by category:
cat_groups = {}
for p in all_active:
    cat = p["category"]
    if cat not in cat_groups:
        cat_groups[cat] = {"count": 0, "mrp": 0.0, "disc": 0.0}
    cat_groups[cat]["count"] += 1
    cat_groups[cat]["mrp"] += p["price"]
    cat_groups[cat]["disc"] += p["discounted_price"]

dept_unweighted_avg_mrp = sum(g["mrp"]/g["count"] for g in cat_groups.values()) / len(cat_groups)
dept_unweighted_avg_disc = sum(g["disc"]/g["count"] for g in cat_groups.values()) / len(cat_groups)

print(f"Unweighted Mean of 10 Department MRP Averages:      Rs. {dept_unweighted_avg_mrp:.2f}")
print(f"Unweighted Mean of 10 Department Selling Averages:  Rs. {dept_unweighted_avg_disc:.2f}")

# Investigate why 10,505 vs --count 10500
# In seed_products.py:
# total_subcategories = 55
# count_per_subcat = int(target_total / total_subcategories) + 1
# if target_total = 10450 or 10500:
# 10500 / 55 = 190.909... -> int is 190 -> 190 + 1 = 191 per subcategory!
# 191 * 55 = 10,505!
print(f"\n--- Seeder Math Analysis ---")
print(f"55 subcategories.")
print(f"int(10500 / 55) = {int(10500 / 55)}")
print(f"count_per_subcat = int(10500 / 55) + 1 = {int(10500 / 55) + 1}")
print(f"Total generated = 55 * {int(10500 / 55) + 1} = {55 * (int(10500 / 55) + 1)}")
print(f"Plus 1 test admin created product = 10,506 total!")


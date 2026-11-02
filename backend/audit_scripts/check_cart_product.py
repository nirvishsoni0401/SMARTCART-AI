import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import products_collection

p = products_collection.find_one({"is_active": True})
print("Sample product keys and values:")
for k in ["product_id", "external_id", "title", "name", "price", "discounted_price", "currency", "stock", "stock_quantity", "images", "thumbnail"]:
    print(f"  {k}: {p.get(k)} (type: {type(p.get(k)).__name__})")

count_with_external_id = products_collection.count_documents({"external_id": {"$exists": True}})
count_with_product_id = products_collection.count_documents({"product_id": {"$exists": True}})
count_currency_inr = products_collection.count_documents({"currency": "INR"})
print(f"Total with product_id: {count_with_product_id}")
print(f"Total with external_id: {count_with_external_id}")
print(f"Total with currency INR: {count_currency_inr}")


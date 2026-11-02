import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import db, client, products_collection, MONGO_URI, LOCAL_MONGO_URI

print(f"Configured MONGO_URI: {MONGO_URI}")
print(f"Configured LOCAL_MONGO_URI: {LOCAL_MONGO_URI}")
print(f"Active Client Address: {client.address}")
print(f"Active Database: {db.name}")
print(f"Total Collections: {db.list_collection_names()}")
print(f"Product Documents Count: {products_collection.count_documents({})}")


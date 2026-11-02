import os
from pymongo import MongoClient

atlas_uri = "mongodb+srv://nirvishsoni2008_db_user:eAjQUbuidj70Q9hh@cluster0.uvw6p79.mongodb.net/"
print("Testing direct connection to Atlas URI...")
try:
    c = MongoClient(atlas_uri, serverSelectionTimeoutMS=4000)
    print("Atlas ping:", c.admin.command('ping'))
    db = c["smartcart_db"]
    print("Atlas collections in smartcart_db:", db.list_collection_names())
    print("Atlas products in smartcart_db:", db["products"].count_documents({}))
    db2 = c["SmartcartAI"]
    print("Atlas collections in SmartcartAI:", db2.list_collection_names())
    print("Atlas products in SmartcartAI:", db2["products"].count_documents({}))
except Exception as e:
    print(f"Atlas direct connection failed: {type(e).__name__}: {e}")


import requests
from app.database import products_collection

DUMMY_API = "https://dummyjson.com/products"

def import_products():

    response = requests.get(
        DUMMY_API,
        params={"limit": 0},
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    products = data["products"]

    for product in products:

        formatted_product = {
            "external_id": product["id"],
            "name": product["title"],
            "description": product["description"],
            "category": product["category"],
            "brand": product.get("brand", "Unknown"),
            "price": product["price"],
            "rating": product["rating"],
            "stock": product["stock"],
            "image_url": product["thumbnail"],
            "images": product.get("images", []),
            "is_active": True
        }

        products_collection.update_one(
            {"external_id": product["id"]},
            {"$set": formatted_product},
            upsert=True
        )

    return len(products)
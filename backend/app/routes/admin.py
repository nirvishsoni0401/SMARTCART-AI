from typing import Optional, Dict, Any, List
import datetime
from fastapi import APIRouter, HTTPException, Path, Body, Query
from pydantic import BaseModel, Field
from app.database import products_collection

router = APIRouter(
    prefix="/api/admin",
    tags=["Admin"]
)


class ProductCreateSchema(BaseModel):
    title: str = Field(..., min_length=2)
    description: str = Field(..., min_length=5)
    brand: str
    category: str
    subcategory: str
    price: float = Field(..., gt=0)
    discounted_price: Optional[float] = None
    discount_percentage: Optional[int] = 0
    stock_quantity: int = Field(50, ge=0)
    images: List[str] = Field(default_factory=list)
    specifications: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tags: Optional[List[str]] = Field(default_factory=list)
    seller: Optional[str] = "SmartCart Verified Retail"


class InventoryUpdateSchema(BaseModel):
    stock_quantity: int = Field(..., ge=0)


@router.get("/stats")
def get_catalog_stats():
    """Returns overview statistics of the large scale catalog"""
    total_products = products_collection.count_documents({"is_active": True})
    total_inactive = products_collection.count_documents({"is_active": False})
    out_of_stock = products_collection.count_documents({"is_active": True, "stock_quantity": {"$lte": 0}})
    low_stock = products_collection.count_documents({"is_active": True, "stock_quantity": {"$gt": 0, "$lte": 10}})

    # Category breakdown
    pipeline = [
        {"$match": {"is_active": True}},
        {"$group": {
            "_id": "$category",
            "count": {"$sum": 1},
            "avg_price": {"$avg": "$discounted_price"},
            "avg_rating": {"$avg": "$rating"}
        }},
        {"$sort": {"count": -1}}
    ]
    category_stats = list(products_collection.aggregate(pipeline))
    formatted_cats = [
        {
            "category": c["_id"],
            "count": c["count"],
            "avg_price": round(c.get("avg_price", 0), 2),
            "avg_rating": round(c.get("avg_rating", 0), 2)
        }
        for c in category_stats
    ]

    return {
        "total_active_products": total_products,
        "total_inactive_products": total_inactive,
        "out_of_stock_count": out_of_stock,
        "low_stock_count": low_stock,
        "category_breakdown": formatted_cats
    }


@router.post("/products")
def create_product(payload: ProductCreateSchema):
    """Add a new product to the catalog"""
    # Generate new product_id
    last_doc = products_collection.find_one({}, sort=[("product_id", -1)])
    next_id = (last_doc.get("product_id", 10000) + 1) if last_doc else 10001

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    disc_price = payload.discounted_price or payload.price
    images = payload.images or ["https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=800&q=80"]

    product_doc = {
        "product_id": next_id,
        "external_id": next_id,
        "title": payload.title,
        "name": payload.title,
        "description": payload.description,
        "brand": payload.brand,
        "category": payload.category,
        "category_slug": payload.category.lower().replace(" & ", "-and-").replace(" ", "-"),
        "subcategory": payload.subcategory,
        "subcategory_slug": payload.subcategory.lower().replace(" ", "-"),
        "price": payload.price,
        "discounted_price": disc_price,
        "discount_percentage": payload.discount_percentage,
        "currency": "INR",
        "rating": 4.5,
        "review_count": 1,
        "stock_quantity": payload.stock_quantity,
        "stock": payload.stock_quantity,
        "availability": "In Stock" if payload.stock_quantity > 10 else ("Low Stock" if payload.stock_quantity > 0 else "Out of Stock"),
        "images": images,
        "thumbnail": images[0],
        "image_url": images[0],
        "seller": payload.seller,
        "specifications": payload.specifications or {},
        "tags": payload.tags or [],
        "popularity_score": 75.0,
        "is_active": True,
        "created_at": now_iso,
        "updated_at": now_iso
    }

    products_collection.insert_one(product_doc)
    product_doc.pop("_id", None)
    return {"message": "Product created successfully", "product": product_doc}


@router.put("/products/{product_id}")
def update_product(product_id: int, updates: Dict[str, Any] = Body(...)):
    """Update fields of an existing product"""
    existing = products_collection.find_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    updates["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Mirror fields if needed
    if "title" in updates:
        updates["name"] = updates["title"]
    if "stock_quantity" in updates:
        updates["stock"] = updates["stock_quantity"]
        updates["availability"] = "In Stock" if updates["stock_quantity"] > 10 else ("Low Stock" if updates["stock_quantity"] > 0 else "Out of Stock")

    products_collection.update_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]}, {"$set": updates})
    updated = products_collection.find_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]})
    updated.pop("_id", None)
    return {"message": "Product updated successfully", "product": updated}


@router.patch("/products/{product_id}/inventory")
def update_inventory(product_id: int, payload: InventoryUpdateSchema):
    """Update stock quantity and availability for a product"""
    existing = products_collection.find_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    new_stock = payload.stock_quantity
    availability = "In Stock" if new_stock > 10 else ("Low Stock" if new_stock > 0 else "Out of Stock")
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    products_collection.update_one(
        {"$or": [{"product_id": product_id}, {"external_id": product_id}]},
        {"$set": {
            "stock_quantity": new_stock,
            "stock": new_stock,
            "availability": availability,
            "updated_at": now_iso
        }}
    )

    return {
        "product_id": product_id,
        "stock_quantity": new_stock,
        "availability": availability,
        "updated_at": now_iso
    }


@router.delete("/products/{product_id}")
def delete_product(product_id: int):
    """Soft delete a product (sets is_active=False)"""
    existing = products_collection.find_one({"$or": [{"product_id": product_id}, {"external_id": product_id}]})
    if not existing:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    products_collection.update_one(
        {"$or": [{"product_id": product_id}, {"external_id": product_id}]},
        {"$set": {"is_active": False, "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}}
    )
    return {"message": f"Product {product_id} deactivated successfully"}


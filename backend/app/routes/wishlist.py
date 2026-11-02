import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel
from app.database import wishlists_collection, products_collection
from app.services.auth_service import get_current_user_optional

router = APIRouter(prefix="/api/wishlist", tags=["Wishlist"])


class WishlistItemAdd(BaseModel):
    product_id: int


def get_wishlist_identifier(request: Request, current_user: Optional[dict]) -> str:
    if current_user and current_user.get("id"):
        return f"user:{current_user['id']}"
    session_id = request.headers.get("x-session-id") or request.cookies.get("smartcart_session_id")
    if not session_id:
        session_id = "guest_default_session"
    return f"session:{session_id}"


def populate_wishlist_items(product_ids: list) -> list:
    items = []
    for pid in product_ids:
        product = products_collection.find_one({"external_id": pid}, {"_id": 0})
        if product:
            items.append({
                "product_id": pid,
                "name": product.get("name"),
                "price": product.get("price"),
                "currency": product.get("currency", "USD"),
                "image_url": product.get("image_url"),
                "category": product.get("category"),
                "brand": product.get("brand"),
                "stock": product.get("stock", 0),
                "in_stock": product.get("stock", 0) > 0,
                "rating": product.get("rating")
            })
    return items


@router.get("/")
def get_wishlist(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_wishlist_identifier(request, current_user)
    wishlist = wishlists_collection.find_one({"user_id": identifier})
    if not wishlist:
        return {"items": [], "total_items": 0}

    product_ids = wishlist.get("product_ids", [])
    populated = populate_wishlist_items(product_ids)
    return {
        "items": populated,
        "total_items": len(populated)
    }


@router.post("/add")
def add_to_wishlist(
    payload: WishlistItemAdd,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_wishlist_identifier(request, current_user)
    product = products_collection.find_one({"external_id": payload.product_id, "is_active": True})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    wishlist = wishlists_collection.find_one({"user_id": identifier})
    
    if not wishlist:
        product_ids = [payload.product_id]
        wishlists_collection.insert_one({
            "user_id": identifier,
            "product_ids": product_ids,
            "created_at": now,
            "updated_at": now
        })
    else:
        product_ids = wishlist.get("product_ids", [])
        if payload.product_id not in product_ids:
            product_ids.append(payload.product_id)
            wishlists_collection.update_one(
                {"user_id": identifier},
                {"$set": {"product_ids": product_ids, "updated_at": now}}
            )

    populated = populate_wishlist_items(product_ids)
    return {
        "message": "Product added to wishlist",
        "items": populated,
        "total_items": len(populated)
    }


@router.delete("/{product_id}")
def remove_from_wishlist(
    product_id: int,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_wishlist_identifier(request, current_user)
    wishlist = wishlists_collection.find_one({"user_id": identifier})
    if not wishlist:
        return {"message": "Wishlist is empty", "items": [], "total_items": 0}

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    product_ids = [p for p in wishlist.get("product_ids", []) if p != product_id]
    
    wishlists_collection.update_one(
        {"user_id": identifier},
        {"$set": {"product_ids": product_ids, "updated_at": now}}
    )

    populated = populate_wishlist_items(product_ids)
    return {
        "message": "Product removed from wishlist",
        "items": populated,
        "total_items": len(populated)
    }


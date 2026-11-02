import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field
from app.database import carts_collection, products_collection
from app.services.auth_service import get_current_user_optional

router = APIRouter(prefix="/api/cart", tags=["Cart"])


class CartItemAdd(BaseModel):
    product_id: int
    quantity: int = Field(1, ge=1)


class CartItemUpdate(BaseModel):
    product_id: int
    quantity: int = Field(..., ge=0)


def get_cart_identifier(request: Request, current_user: Optional[dict]) -> str:
    """Return user_id if logged in, otherwise session_id or guest header"""
    if current_user and current_user.get("id"):
        return f"user:{current_user['id']}"
    
    session_id = request.headers.get("x-session-id") or request.cookies.get("smartcart_session_id")
    if not session_id:
        session_id = "guest_default_session"
    return f"session:{session_id}"


def populate_cart_items(items: list) -> list:
    populated = []
    for item in items:
        prod_id = item["product_id"]
        product = products_collection.find_one({"external_id": prod_id}, {"_id": 0})
        if product:
            available_stock = product.get("stock", 0)
            qty = min(item["quantity"], max(1, available_stock))
            price = float(product.get("price", 0))
            populated.append({
                "product_id": prod_id,
                "quantity": qty,
                "name": product.get("name", "Unknown Product"),
                "price": price,
                "currency": product.get("currency", "USD"),
                "image_url": product.get("image_url", ""),
                "category": product.get("category", ""),
                "brand": product.get("brand", "SmartCart"),
                "stock": available_stock,
                "in_stock": available_stock > 0,
                "total_price": round(price * qty, 2)
            })
    return populated


@router.get("/")
def get_cart(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_cart_identifier(request, current_user)
    cart = carts_collection.find_one({"user_id": identifier})
    if not cart:
        return {
            "items": [],
            "total_items": 0,
            "subtotal": 0.0,
            "currency": "USD"
        }

    populated_items = populate_cart_items(cart.get("items", []))
    subtotal = round(sum(i["total_price"] for i in populated_items), 2)
    total_items = sum(i["quantity"] for i in populated_items)

    return {
        "items": populated_items,
        "total_items": total_items,
        "subtotal": subtotal,
        "currency": "USD"
    }


@router.post("/add")
def add_to_cart(
    payload: CartItemAdd,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_cart_identifier(request, current_user)
    
    # Check product existence & stock
    product = products_collection.find_one({"external_id": payload.product_id, "is_active": True})
    if not product:
        raise HTTPException(status_code=404, detail="Product not found or unavailable")
    
    if product.get("stock", 0) <= 0:
        raise HTTPException(status_code=400, detail="Product is out of stock")

    cart = carts_collection.find_one({"user_id": identifier})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if not cart:
        items = [{"product_id": payload.product_id, "quantity": payload.quantity}]
        carts_collection.insert_one({
            "user_id": identifier,
            "items": items,
            "created_at": now,
            "updated_at": now
        })
    else:
        items = cart.get("items", [])
        found = False
        for item in items:
            if item["product_id"] == payload.product_id:
                new_qty = item["quantity"] + payload.quantity
                if new_qty > product.get("stock", 0):
                    item["quantity"] = product.get("stock", 0)
                else:
                    item["quantity"] = new_qty
                found = True
                break
        if not found:
            items.append({"product_id": payload.product_id, "quantity": min(payload.quantity, product.get("stock", 0))})
            
        carts_collection.update_one(
            {"user_id": identifier},
            {"$set": {"items": items, "updated_at": now}}
        )

    populated = populate_cart_items(items)
    return {
        "message": "Item added to cart",
        "items": populated,
        "total_items": sum(i["quantity"] for i in populated),
        "subtotal": round(sum(i["total_price"] for i in populated), 2)
    }


@router.put("/update")
def update_cart_item(
    payload: CartItemUpdate,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_cart_identifier(request, current_user)
    cart = carts_collection.find_one({"user_id": identifier})
    if not cart:
        raise HTTPException(status_code=404, detail="Cart not found")

    items = cart.get("items", [])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if payload.quantity <= 0:
        items = [i for i in items if i["product_id"] != payload.product_id]
    else:
        product = products_collection.find_one({"external_id": payload.product_id})
        max_stock = product.get("stock", 999) if product else 999
        updated = False
        for item in items:
            if item["product_id"] == payload.product_id:
                item["quantity"] = min(payload.quantity, max_stock)
                updated = True
                break
        if not updated:
            items.append({"product_id": payload.product_id, "quantity": min(payload.quantity, max_stock)})

    carts_collection.update_one(
        {"user_id": identifier},
        {"$set": {"items": items, "updated_at": now}}
    )

    populated = populate_cart_items(items)
    return {
        "message": "Cart updated",
        "items": populated,
        "total_items": sum(i["quantity"] for i in populated),
        "subtotal": round(sum(i["total_price"] for i in populated), 2)
    }


@router.delete("/{product_id}")
def remove_from_cart(
    product_id: int,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_cart_identifier(request, current_user)
    cart = carts_collection.find_one({"user_id": identifier})
    if not cart:
        return {"message": "Cart already empty", "items": [], "total_items": 0, "subtotal": 0.0}

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    items = [i for i in cart.get("items", []) if i["product_id"] != product_id]

    carts_collection.update_one(
        {"user_id": identifier},
        {"$set": {"items": items, "updated_at": now}}
    )

    populated = populate_cart_items(items)
    return {
        "message": "Item removed from cart",
        "items": populated,
        "total_items": sum(i["quantity"] for i in populated),
        "subtotal": round(sum(i["total_price"] for i in populated), 2)
    }


@router.delete("/")
def clear_cart(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    identifier = get_cart_identifier(request, current_user)
    carts_collection.delete_one({"user_id": identifier})
    return {"message": "Cart cleared", "items": [], "total_items": 0, "subtotal": 0.0}


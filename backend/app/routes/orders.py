import datetime
import uuid
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field, EmailStr
from app.database import orders_collection, products_collection, carts_collection
from app.services.auth_service import get_current_user_optional

router = APIRouter(prefix="/api/orders", tags=["Orders"])


class ShippingAddress(BaseModel):
    full_name: str
    street_address: str
    city: str
    state: str
    postal_code: str
    country: str = "United States"
    phone: Optional[str] = None


class CheckoutOrderItem(BaseModel):
    product_id: int
    quantity: int = Field(..., ge=1)


class CreateOrderRequest(BaseModel):
    items: Optional[List[CheckoutOrderItem]] = None
    shipping_address: ShippingAddress
    customer_email: Optional[EmailStr] = None
    customer_name: Optional[str] = None
    payment_method: str = "card_mock"
    notes: Optional[str] = None


@router.post("/", status_code=status.HTTP_201_CREATED)
def create_order(
    req: CreateOrderRequest,
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    user_id = current_user.get("id") if current_user else None
    email = current_user.get("email") if current_user else req.customer_email
    name = current_user.get("name") if current_user else req.customer_name

    if not email:
        raise HTTPException(status_code=400, detail="Customer email is required for checkout")

    # If items are not provided explicitly in request body, retrieve from cart
    items_to_order = req.items
    cart_identifier = f"user:{user_id}" if user_id else f"session:{request.headers.get('x-session-id', 'guest_default_session')}"

    if not items_to_order:
        cart = carts_collection.find_one({"user_id": cart_identifier})
        if not cart or not cart.get("items"):
            raise HTTPException(status_code=400, detail="Cart is empty")
        items_to_order = [CheckoutOrderItem(product_id=i["product_id"], quantity=i["quantity"]) for i in cart["items"]]

    if not items_to_order:
        raise HTTPException(status_code=400, detail="No items to place order")

    # Step 1: Validate stock for all items
    order_items_detailed = []
    subtotal = 0.0

    for item in items_to_order:
        product = products_collection.find_one({"external_id": item.product_id, "is_active": True})
        if not product:
            raise HTTPException(status_code=400, detail=f"Product with ID {item.product_id} is unavailable")
        
        current_stock = product.get("stock", 0)
        if current_stock < item.quantity:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient stock for '{product.get('name')}'. Available: {current_stock}, Requested: {item.quantity}"
            )
        
        price = float(product.get("price", 0))
        item_total = round(price * item.quantity, 2)
        subtotal += item_total

        order_items_detailed.append({
            "product_id": item.product_id,
            "name": product.get("name"),
            "category": product.get("category"),
            "brand": product.get("brand"),
            "price": price,
            "quantity": item.quantity,
            "image_url": product.get("image_url"),
            "total_price": item_total
        })

    # Step 2: Calculate shipping, tax, total
    shipping_fee = 0.0 if subtotal >= 75.0 else 9.99
    tax = round(subtotal * 0.08, 2)  # 8% estimated tax
    total_amount = round(subtotal + shipping_fee + tax, 2)

    # Step 3: Deduct stock atomically
    for item in items_to_order:
        products_collection.update_one(
            {"external_id": item.product_id},
            {"$inc": {"stock": -item.quantity}}
        )

    order_id = f"SC-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    order_doc = {
        "order_id": order_id,
        "user_id": user_id,
        "customer_email": email,
        "customer_name": name or "Guest Shopper",
        "items": order_items_detailed,
        "subtotal": round(subtotal, 2),
        "shipping_fee": shipping_fee,
        "tax": tax,
        "total_amount": total_amount,
        "currency": "USD",
        "shipping_address": req.shipping_address.model_dump(),
        "payment_method": req.payment_method,
        "payment_status": "paid",
        "order_status": "confirmed",
        "created_at": now_iso,
        "updated_at": now_iso
    }

    orders_collection.insert_one(order_doc)
    order_doc.pop("_id", None)

    # Step 4: Clear user's cart
    carts_collection.delete_one({"user_id": cart_identifier})

    return {
        "message": "Order placed successfully",
        "order": order_doc
    }


@router.get("/")
def get_orders(
    request: Request,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    query = {}
    if current_user and current_user.get("id"):
        query = {"$or": [{"user_id": current_user["id"]}, {"customer_email": current_user.get("email")}]}
    else:
        # Check session / guest email
        email = request.query_params.get("email")
        if email:
            query = {"customer_email": email.strip().lower()}
        else:
            return {"orders": [], "total": 0}

    orders = list(orders_collection.find(query, {"_id": 0}).sort("created_at", -1))
    return {
        "orders": orders,
        "total": len(orders)
    }


@router.get("/{order_id}")
def get_order_by_id(order_id: str):
    order = orders_collection.find_one({"order_id": order_id}, {"_id": 0})
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {"order": order}


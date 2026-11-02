from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import CartItem, Product, User, InteractionEvent
from app.schemas import CartItemAdd, CartItemUpdate, CartItemOut, CartOut, ProductOut
from app.auth import get_current_user_optional
from app.config import settings

router = APIRouter(prefix="/cart", tags=["Cart"])

def get_cart_filter(user: Optional[User], session_id: Optional[str]):
    if user:
        return CartItem.user_id == user.id
    if session_id:
        return CartItem.session_id == session_id
    return None

@router.get("", response_model=CartOut)
def get_cart(
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    cart_filter = get_cart_filter(current_user, session_id)
    if cart_filter is None:
        return CartOut(items=[], total_items=0, subtotal=0.0)

    items = db.query(CartItem).filter(cart_filter).all()
    out_items = []
    subtotal = 0.0
    total_qty = 0

    for item in items:
        item_total = item.product.price * item.quantity
        subtotal += item_total
        total_qty += item.quantity
        out_items.append(
            CartItemOut(
                id=item.id,
                product_id=item.product_id,
                quantity=item.quantity,
                price=item.product.price,
                product=ProductOut.model_validate(item.product)
            )
        )

    return CartOut(
        items=out_items,
        total_items=total_qty,
        subtotal=round(subtotal, 2)
    )

@router.post("", response_model=CartOut, status_code=status.HTTP_201_CREATED)
def add_to_cart(
    item_in: CartItemAdd,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == item_in.product_id).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found")

    cart_filter = get_cart_filter(current_user, item_in.session_id)
    if cart_filter is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide authentication or session_id")

    existing = db.query(CartItem).filter(
        cart_filter,
        CartItem.product_id == item_in.product_id
    ).first()

    if existing:
        existing.quantity += item_in.quantity
    else:
        new_item = CartItem(
            user_id=current_user.id if current_user else None,
            session_id=item_in.session_id if not current_user else None,
            product_id=item_in.product_id,
            quantity=item_in.quantity
        )
        db.add(new_item)

    # Automatically record cart_add event for recommender
    should_track = current_user.consent_tracking if current_user else True
    if should_track:
        algo = current_user.assigned_algorithm if current_user else settings.GLOBAL_ALGORITHM
        event = InteractionEvent(
            user_id=current_user.id if current_user else None,
            session_id=item_in.session_id,
            product_id=item_in.product_id,
            event_type="cart_add",
            reward=settings.rewards.cart_add,
            algorithm=algo
        )
        db.add(event)

    db.commit()
    return get_cart(session_id=item_in.session_id, current_user=current_user, db=db)

@router.put("/{item_id}", response_model=CartOut)
def update_cart_item(
    item_id: int,
    item_in: CartItemUpdate,
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    cart_filter = get_cart_filter(current_user, session_id)
    item = db.query(CartItem).filter(CartItem.id == item_id, cart_filter).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart item not found")

    if item_in.quantity <= 0:
        db.delete(item)
    else:
        item.quantity = item_in.quantity

    db.commit()
    return get_cart(session_id=session_id, current_user=current_user, db=db)

@router.delete("/{item_id}", response_model=CartOut)
def remove_from_cart(
    item_id: int,
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    cart_filter = get_cart_filter(current_user, session_id)
    item = db.query(CartItem).filter(CartItem.id == item_id, cart_filter).first()
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Cart item not found")

    prod_id = item.product_id
    db.delete(item)

    # Record cart_remove event
    should_track = current_user.consent_tracking if current_user else True
    if should_track:
        algo = current_user.assigned_algorithm if current_user else settings.GLOBAL_ALGORITHM
        event = InteractionEvent(
            user_id=current_user.id if current_user else None,
            session_id=session_id,
            product_id=prod_id,
            event_type="cart_remove",
            reward=settings.rewards.cart_remove,
            algorithm=algo
        )
        db.add(event)

    db.commit()
    return get_cart(session_id=session_id, current_user=current_user, db=db)

@router.delete("", response_model=CartOut)
def clear_cart(
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    cart_filter = get_cart_filter(current_user, session_id)
    if cart_filter is not None:
        db.query(CartItem).filter(cart_filter).delete()
        db.commit()
    return CartOut(items=[], total_items=0, subtotal=0.0)

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import WishlistItem, Product, User, InteractionEvent
from app.schemas import WishlistAdd, WishlistItemOut, WishlistOut, ProductOut
from app.auth import get_current_user_optional
from app.config import settings

router = APIRouter(prefix="/wishlist", tags=["Wishlist"])

def get_wishlist_filter(user: Optional[User], session_id: Optional[str]):
    if user:
        return WishlistItem.user_id == user.id
    if session_id:
        return WishlistItem.session_id == session_id
    return None

@router.get("", response_model=WishlistOut)
def get_wishlist(
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    w_filter = get_wishlist_filter(current_user, session_id)
    if w_filter is None:
        return WishlistOut(items=[], total_items=0)

    items = db.query(WishlistItem).filter(w_filter).all()
    out_items = [
        WishlistItemOut(
            id=item.id,
            product_id=item.product_id,
            product=ProductOut.model_validate(item.product),
            created_at=item.created_at
        )
        for item in items
    ]
    return WishlistOut(items=out_items, total_items=len(out_items))

@router.post("", response_model=WishlistOut, status_code=status.HTTP_201_CREATED)
def add_to_wishlist(
    item_in: WishlistAdd,
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    product = db.query(Product).filter(Product.id == item_in.product_id, Product.is_active == True).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found or inactive")

    w_filter = get_wishlist_filter(current_user, item_in.session_id)
    if w_filter is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide authentication or session_id")

    existing = db.query(WishlistItem).filter(
        w_filter,
        WishlistItem.product_id == item_in.product_id
    ).first()

    if not existing:
        new_item = WishlistItem(
            user_id=current_user.id if current_user else None,
            session_id=item_in.session_id if not current_user else None,
            product_id=item_in.product_id
        )
        db.add(new_item)

        # Log wishlist event (+3 reward)
        should_track = current_user.consent_tracking if current_user else True
        if should_track:
            algo = current_user.assigned_algorithm if current_user else settings.GLOBAL_ALGORITHM
            event = InteractionEvent(
                user_id=current_user.id if current_user else None,
                session_id=item_in.session_id,
                product_id=item_in.product_id,
                event_type="wishlist",
                reward=settings.rewards.wishlist,
                algorithm=algo,
                origin="live"
            )
            db.add(event)

        db.commit()

    return get_wishlist(session_id=item_in.session_id, current_user=current_user, db=db)

@router.delete("/{product_id}", response_model=WishlistOut)
def remove_from_wishlist(
    product_id: int,
    session_id: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db)
):
    w_filter = get_wishlist_filter(current_user, session_id)
    if w_filter is None:
        return WishlistOut(items=[], total_items=0)

    db.query(WishlistItem).filter(
        w_filter,
        WishlistItem.product_id == product_id
    ).delete()
    db.commit()

    return get_wishlist(session_id=session_id, current_user=current_user, db=db)

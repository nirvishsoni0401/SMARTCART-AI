from datetime import datetime, timezone
from typing import List
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Order, OrderItem, CartItem, Product, User, InteractionEvent, RecommendationLog
from app.schemas import OrderCreate, OrderOut, OrderItemOut, ProductOut, ReturnItemRequest
from app.auth import get_current_user
from app.config import settings
from app.services.model_store import model_store
from app.services.feature_engineering import build_user_context, extract_feature_vector

router = APIRouter(prefix="/orders", tags=["Orders"])

@router.post("", response_model=OrderOut, status_code=status.HTTP_201_CREATED)
def checkout(
    order_in: OrderCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Fetch user's cart items
    cart_items = db.query(CartItem).filter(CartItem.user_id == current_user.id).all()
    if not cart_items:
        # Fallback to session items if user logged in during checkout
        if order_in.session_id:
            cart_items = db.query(CartItem).filter(CartItem.session_id == order_in.session_id).all()

    if not cart_items:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Your cart is empty")

    # Ensure all cart products are active
    for item in cart_items:
        if not item.product.is_active:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Product '{item.product.name}' is no longer active.")

    total_amount = sum(item.product.price * item.quantity for item in cart_items)

    order = Order(
        user_id=current_user.id,
        total_amount=round(total_amount, 2),
        shipping_address=order_in.shipping_address,
        payment_method=order_in.payment_method or "UPI / Mock Pay",
        status="placed"
    )
    db.add(order)
    db.flush()  # assign order.id

    algo = current_user.assigned_algorithm or settings.GLOBAL_ALGORITHM
    should_track = current_user.consent_tracking

    for item in cart_items:
        order_item = OrderItem(
            order_id=order.id,
            product_id=item.product_id,
            quantity=item.quantity,
            price=item.product.price,
            status="placed"
        )
        db.add(order_item)

        # Decrement stock
        if item.product.stock >= item.quantity:
            item.product.stock -= item.quantity

        # Record purchase event (+10 reward)
        if should_track:
            # Look up recent recommendation log if exists
            recent_log = db.query(RecommendationLog).filter(
                RecommendationLog.user_id == current_user.id,
                RecommendationLog.product_id == item.product_id
            ).order_by(RecommendationLog.created_at.desc()).first()

            event = InteractionEvent(
                user_id=current_user.id,
                session_id=order_in.session_id,
                product_id=item.product_id,
                event_type="purchase",
                reward=settings.rewards.purchase,
                algorithm=algo,
                origin="live",
                recommendation_log_id=recent_log.id if recent_log else None
            )
            db.add(event)

    # Clear cart
    db.query(CartItem).filter(
        (CartItem.user_id == current_user.id) |
        (CartItem.session_id == order_in.session_id)
    ).delete(synchronize_session=False)

    db.commit()
    db.refresh(order)
    return OrderOut.model_validate(order)

@router.get("", response_model=List[OrderOut])
def get_orders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    orders = db.query(Order).filter(Order.user_id == current_user.id).order_by(Order.created_at.desc()).all()
    return [OrderOut.model_validate(o) for o in orders]

@router.get("/{order_id}", response_model=OrderOut)
def get_order(
    order_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    order = db.query(Order).filter(Order.id == order_id, Order.user_id == current_user.id).first()
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")
    return OrderOut.model_validate(order)

@router.post("/{order_id}/items/{item_id}/return", response_model=OrderOut)
def return_order_item(
    order_id: int,
    item_id: int,
    return_req: ReturnItemRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    order = db.query(Order).filter(Order.id == order_id, Order.user_id == current_user.id).first()
    if not order:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order not found")

    order_item = db.query(OrderItem).filter(OrderItem.id == item_id, OrderItem.order_id == order_id).first()
    if not order_item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Order item not found")

    if order_item.status == "returned":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Item has already been returned")

    order_item.status = "returned"
    order_item.returned_at = datetime.now(timezone.utc)

    # Restore stock
    order_item.product.stock += order_item.quantity

    # Emit delayed return event and update bandit via correct_reward (A & A_inv unchanged)
    should_track = current_user.consent_tracking
    if should_track:
        recent_log = db.query(RecommendationLog).filter(
            RecommendationLog.user_id == current_user.id,
            RecommendationLog.product_id == order_item.product_id
        ).order_by(RecommendationLog.created_at.desc()).first()

        algo = recent_log.algorithm if recent_log else (current_user.assigned_algorithm or settings.GLOBAL_ALGORITHM)
        
        # Delta reward: return reward minus purchase reward (-8.0 - 10.0 = -18.0)
        # Ensuring net reward equals return_item reward (-8.0)
        delta_reward = settings.rewards.return_item - settings.rewards.purchase

        # 0.2: Use ONLY the stored feature-vector snapshot from original log; NEVER recompute on-the-fly.
        stored_vector = None
        if recent_log and (recent_log.feature_vector_snapshot or recent_log.feature_vector):
            stored_vector = recent_log.feature_vector_snapshot or recent_log.feature_vector

        if stored_vector is not None:
            feature_vector = np.array(stored_vector, dtype=np.float64)
            # Apply correction to model reward vector b WITHOUT modifying A or A_inv
            model = model_store.get_model(algo)
            if model is not None:
                model.correct_reward(feature_vector, delta_reward)
                model_store.save_model_to_db(algo, db=db)
            else:
                import logging
                logging.getLogger(__name__).warning(
                    f"Skipping return correction: algorithm model '{algo}' not found."
                )
        else:
            import logging
            logging.getLogger(__name__).warning(
                f"Skipping return correction for order {order_id} item {item_id}: "
                f"no recommendation log with stored feature vector found."
            )

        event = InteractionEvent(
            user_id=current_user.id,
            product_id=order_item.product_id,
            event_type="return",
            reward=settings.rewards.return_item,
            algorithm=algo,
            origin="live",
            recommendation_log_id=recent_log.id if recent_log else None
        )
        db.add(event)

    db.commit()
    db.refresh(order)
    return OrderOut.model_validate(order)

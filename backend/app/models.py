from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
)
from sqlalchemy.orm import relationship
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user", nullable=False)  # "user" or "admin"
    assigned_algorithm = Column(String(50), default="linucb", nullable=False)
    consent_tracking = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    orders = relationship("Order", back_populates="user", cascade="all, delete-orphan")
    cart_items = relationship("CartItem", back_populates="user", cascade="all, delete-orphan")
    wishlist_items = relationship("WishlistItem", back_populates="user", cascade="all, delete-orphan")


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    brand = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=False, index=True)  # Selling price in INR
    mrp = Column(Float, nullable=False)                # Maximum Retail Price in INR
    discount_pct = Column(Float, default=0.0, nullable=False)
    rating = Column(Float, default=4.0, nullable=False, index=True)
    rating_count = Column(Integer, default=0, nullable=False)
    popularity_score = Column(Float, default=50.0, nullable=False, index=True)
    popularity_prior = Column(Float, default=0.0, nullable=True)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=False)
    attributes = Column(JSON, default=dict, nullable=False)
    stock = Column(Integer, default=100, nullable=False)
    
    # Phase 8 Catalog Integration fields
    source = Column(String(50), default="seed", nullable=False, index=True)  # "dummyjson" or "seed"
    source_id = Column(String(100), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    raw_price_usd = Column(Float, nullable=True)
    imported_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    reviews = relationship("ProductReview", back_populates="product", cascade="all, delete-orphan")


class ProductReview(Base):
    __tablename__ = "product_reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False, index=True)
    reviewer_name = Column(String(100), nullable=False)
    rating = Column(Float, nullable=False)
    comment = Column(Text, nullable=False)
    date = Column(DateTime, nullable=True)
    source = Column(String(50), default="dummyjson", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    product = relationship("Product", back_populates="reviews")


class CartItem(Base):
    __tablename__ = "cart_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="cart_items")
    product = relationship("Product")


class WishlistItem(Base):
    __tablename__ = "wishlist_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="wishlist_items")
    product = relationship("Product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    total_amount = Column(Float, nullable=False)
    shipping_address = Column(JSON, default=dict, nullable=False)
    payment_method = Column(String(50), default="UPI / Mock Pay", nullable=False)
    status = Column(String(50), default="placed", nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, default=1, nullable=False)
    price = Column(Float, nullable=False)
    status = Column(String(50), default="placed", nullable=False)  # "placed" or "returned"
    returned_at = Column(DateTime, nullable=True)

    order = relationship("Order", back_populates="items")
    product = relationship("Product")


class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    recommendation_id = Column(String(50), nullable=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    algorithm = Column(String(50), nullable=False, index=True)
    model_version = Column(String(50), default="v2.0", nullable=False, index=True)
    feature_vector = Column(JSON, nullable=False)  # List[float] representing phi(u, p)
    feature_vector_snapshot = Column(JSON, nullable=True)
    score = Column(Float, nullable=False)
    ucb_bonus = Column(Float, default=0.0, nullable=False)
    was_exploration = Column(Boolean, default=False, nullable=False)
    rank = Column(Integer, nullable=False)
    ranked_list = Column(JSON, nullable=True)
    origin = Column(String(20), default="live", nullable=False, index=True)  # "live" | "simulated" | "legacy"
    resolved = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    product = relationship("Product")


class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, nullable=True, index=True)
    session_id = Column(String(100), nullable=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    event_type = Column(String(50), nullable=False, index=True)
    reward = Column(Float, nullable=False)
    algorithm = Column(String(50), nullable=False, index=True)
    recommendation_log_id = Column(Integer, ForeignKey("recommendation_logs.id"), nullable=True, index=True)
    origin = Column(String(20), default="live", nullable=False, index=True)  # "live" | "simulated" | "legacy"
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)

    product = relationship("Product")
    recommendation_log = relationship("RecommendationLog")


class ModelState(Base):
    __tablename__ = "model_states"

    algorithm = Column(String(50), primary_key=True, index=True)
    state_blob = Column(Text, nullable=False)  # Serialized model parameters / JSON
    n_updates = Column(Integer, default=0, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)


class ModelHistory(Base):
    __tablename__ = "model_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    algorithm = Column(String(50), nullable=False, index=True)
    action = Column(String(50), default="apply_warm_start", nullable=False)
    snapshot_key = Column(String(100), nullable=False)
    origin = Column(String(20), default="warm_start", nullable=False)
    metadata_json = Column(JSON, nullable=True)
    applied_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False, index=True)


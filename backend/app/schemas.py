from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, EmailStr, Field

# --- User & Auth Schemas ---

class UserRegister(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    role: Optional[str] = "user"
    consent_tracking: Optional[bool] = True

UserCreate = UserRegister

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    email: str
    role: str
    assigned_algorithm: str
    consent_tracking: bool
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut

class ConsentUpdate(BaseModel):
    consent_tracking: bool

# --- Product Schemas ---

class ProductReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    reviewer_name: str
    rating: float
    comment: str
    date: Optional[datetime] = None
    source: str = "dummyjson"
    created_at: datetime

class ProductOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    category: str
    brand: str
    price: float
    mrp: float
    discount_pct: float
    rating: float
    rating_count: int
    popularity_score: float
    description: Optional[str] = None
    image_url: str
    attributes: Dict[str, Any] = {}
    stock: int
    source: str = "seed"
    source_id: Optional[str] = None
    is_active: bool = True
    raw_price_usd: Optional[float] = None
    imported_at: Optional[datetime] = None
    created_at: datetime
    reviews: List[ProductReviewOut] = []

class PaginatedProducts(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    products: List[ProductOut]

# --- Cart Schemas ---

class CartItemAdd(BaseModel):
    product_id: int
    quantity: int = Field(1, ge=1)
    session_id: Optional[str] = None

class CartItemUpdate(BaseModel):
    quantity: int = Field(..., ge=1)

class CartItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    quantity: int
    price: Optional[float] = None
    product: ProductOut
    created_at: Optional[datetime] = None

class CartOut(BaseModel):
    items: List[CartItemOut]
    total_items: int
    subtotal: float

# --- Wishlist Schemas ---

class WishlistItemAdd(BaseModel):
    product_id: int
    session_id: Optional[str] = None

WishlistAdd = WishlistItemAdd

class WishlistItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    product: ProductOut
    created_at: datetime

class WishlistOut(BaseModel):
    items: List[WishlistItemOut]
    total_items: int

# --- Order Schemas ---

class OrderCreate(BaseModel):
    shipping_address: Dict[str, Any]
    payment_method: Optional[str] = "UPI / Mock Pay"
    session_id: Optional[str] = None

class OrderItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    quantity: int
    price: float
    status: str
    returned_at: Optional[datetime] = None
    product: ProductOut

class OrderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    total_amount: float
    shipping_address: Dict[str, Any]
    payment_method: str
    status: str
    created_at: datetime
    items: List[OrderItemOut]

class ReturnItemRequest(BaseModel):
    reason: Optional[str] = "Not satisfied"

# --- Recommendation Schemas ---

class RecommendationItemOut(BaseModel):
    product: ProductOut
    score: float
    ucb_bonus: float
    was_exploration: bool
    explanation: str
    algorithm: str
    log_id: Optional[int] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItemOut]
    algorithm: str
    cold_start: bool
    latency_ms: float
    session_id: Optional[str] = None

# --- Event Schemas ---

class EventCreate(BaseModel):
    product_id: int
    event_type: str  # click, dislike, wishlist, cart_add, cart_remove, purchase, return, skip
    recommendation_log_id: Optional[int] = None
    session_id: Optional[str] = None
    origin: Optional[str] = "live"

class ImpressionBatchCreate(BaseModel):
    product_ids: List[int]
    recommendation_log_ids: Optional[List[int]] = None
    session_id: Optional[str] = None
    origin: Optional[str] = "live"

class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    product_id: int
    event_type: str
    reward: float
    algorithm: str
    recommendation_log_id: Optional[int]
    origin: str = "live"
    created_at: datetime

# --- Admin Schemas ---

class AlgorithmUpdateRequest(BaseModel):
    algorithm: str

class RewardConfigUpdate(BaseModel):
    purchase: Optional[float] = None
    cart_add: Optional[float] = None
    wishlist: Optional[float] = None
    click: Optional[float] = None
    skip: Optional[float] = None
    cart_remove: Optional[float] = None
    dislike: Optional[float] = None
    return_item: Optional[float] = None

class SimulateRequest(BaseModel):
    num_rounds: int = Field(50, ge=1, le=500)
    algorithms: Optional[List[str]] = None
    train_live: bool = False
    origin: Optional[str] = "simulated"

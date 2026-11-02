from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc, asc, func
from app.database import get_db
from app.models import Product
from app.schemas import ProductOut, PaginatedProducts

router = APIRouter(prefix="/products", tags=["Products"])

@router.get("", response_model=PaginatedProducts)
def list_products(
    q: Optional[str] = Query(None, description="Search keyword in title/brand/desc"),
    category: Optional[str] = Query(None, description="Category filter"),
    brand: Optional[str] = Query(None, description="Brand filter"),
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_rating: Optional[float] = Query(None, ge=0, le=5),
    min_discount: Optional[float] = Query(None, ge=0, le=100),
    in_stock: Optional[bool] = Query(None),
    sort_by: Optional[str] = Query("popularity", description="popularity, price_asc, price_desc, rating_desc, newest"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    # Enforce active products only
    query = db.query(Product).filter(Product.is_active == True)

    if q:
        clean_q = f"%{q.strip()}%"
        query = query.filter(
            or_(
                Product.name.ilike(clean_q),
                Product.brand.ilike(clean_q),
                Product.category.ilike(clean_q),
                Product.description.ilike(clean_q)
            )
        )

    if category and category.lower() != "all":
        query = query.filter(Product.category.ilike(category))

    if brand and brand.lower() != "all":
        query = query.filter(Product.brand.ilike(brand))

    if min_price is not None:
        query = query.filter(Product.price >= min_price)

    if max_price is not None:
        query = query.filter(Product.price <= max_price)

    if min_rating is not None:
        query = query.filter(Product.rating >= min_rating)

    if min_discount is not None:
        query = query.filter(Product.discount_pct >= min_discount)

    if in_stock is True:
        query = query.filter(Product.stock > 0)

    # Sorting
    if sort_by == "price_asc":
        query = query.order_by(asc(Product.price))
    elif sort_by == "price_desc":
        query = query.order_by(desc(Product.price))
    elif sort_by == "rating_desc":
        query = query.order_by(desc(Product.rating), desc(Product.rating_count))
    elif sort_by == "newest":
        query = query.order_by(desc(Product.created_at))
    else:  # "popularity" (default)
        query = query.order_by(desc(Product.popularity_score), desc(Product.rating))

    total = query.count()
    total_pages = max(1, (total + page_size - 1) // page_size)
    offset = (page - 1) * page_size
    products = query.offset(offset).limit(page_size).all()

    return PaginatedProducts(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        products=[ProductOut.model_validate(p) for p in products]
    )

@router.get("/categories")
def get_categories(db: Session = Depends(get_db)):
    results = db.query(
        Product.category,
        func.count(Product.id).label("count")
    ).filter(Product.is_active == True).group_by(Product.category).order_by(desc("count")).all()
    return [{"name": r[0], "count": r[1]} for r in results]

@router.get("/brands")
def get_brands(category: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Product.brand).filter(Product.is_active == True)
    if category and category.lower() != "all":
        query = query.filter(Product.category.ilike(category))
    brands = [r[0] for r in query.distinct().order_by(Product.brand).all() if r[0]]
    return brands

@router.get("/{product_id}", response_model=ProductOut)
def get_product(product_id: int, db: Session = Depends(get_db)):
    product = db.query(Product).filter(Product.id == product_id, Product.is_active == True).first()
    if not product:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found or inactive")
    return ProductOut.model_validate(product)

@router.get("/{product_id}/similar", response_model=List[ProductOut])
def get_similar_products(product_id: int, limit: int = 6, db: Session = Depends(get_db)):
    target = db.query(Product).filter(Product.id == product_id, Product.is_active == True).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found or inactive")

    similar = db.query(Product).filter(
        Product.category == target.category,
        Product.id != target.id,
        Product.is_active == True,
        Product.stock > 0
    ).order_by(
        desc(Product.rating),
        desc(Product.popularity_score)
    ).limit(limit).all()

    return [ProductOut.model_validate(p) for p in similar]

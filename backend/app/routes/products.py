from typing import Optional, List
from fastapi import APIRouter, Query, HTTPException, Path
from bson import ObjectId
from app.database import products_collection

router = APIRouter(
    prefix="/api/products",
    tags=["Products"]
)


def format_product_doc(doc: dict) -> dict:
    """Ensure consistent product response schema with backward and forward compatibility"""
    doc.pop("_id", None)
    
    # Ensure ID compatibility
    p_id = doc.get("product_id") or doc.get("external_id")
    if p_id is not None:
        doc["product_id"] = p_id
        doc["external_id"] = p_id
        
    # Ensure title / name compatibility
    title = doc.get("title") or doc.get("name", "Product")
    doc["title"] = title
    doc["name"] = title

    # Ensure currency is INR
    doc["currency"] = "INR"

    # Ensure price and discounted price
    price = doc.get("price", 0.0)
    disc_price = doc.get("discounted_price", price)
    doc["price"] = price
    doc["discounted_price"] = disc_price
    
    # Stock compatibility
    stock = doc.get("stock_quantity", doc.get("stock", 0))
    doc["stock_quantity"] = stock
    doc["stock"] = stock
    
    # Availability
    if "availability" not in doc:
        doc["availability"] = "In Stock" if stock > 10 else ("Low Stock" if stock > 0 else "Out of Stock")

    # Images compatibility
    if "images" not in doc or not doc["images"]:
        fallback_img = doc.get("image_url") or doc.get("thumbnail") or "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=800&q=80"
        doc["images"] = [fallback_img]
    
    doc["thumbnail"] = doc["images"][0]
    doc["image_url"] = doc["images"][0]

    return doc


@router.get("/")
def get_products(
    category: Optional[str] = None,
    subcategory: Optional[str] = None,
    brand: Optional[str] = None,
    min_price: Optional[float] = Query(None, ge=0),
    max_price: Optional[float] = Query(None, ge=0),
    min_rating: Optional[float] = Query(None, ge=0, le=5),
    min_discount: Optional[int] = Query(None, ge=0, le=90),
    in_stock_only: Optional[bool] = False,
    sort_by: Optional[str] = Query(
        "default",
        description="default, price_asc, price_desc, rating_desc, popularity, newest, discount_desc"
    ),
    limit: int = Query(24, ge=1, le=100),
    skip: int = Query(0, ge=0),
    page: Optional[int] = Query(None, ge=1)
):
    """
    Search and filter products with server-side pagination and multiple sort criteria.
    """
    if page is not None:
        skip = (page - 1) * limit

    query = {"is_active": True}

    if category and category.lower() != "all":
        # Match case-insensitively or by category_slug
        query["$or"] = [
            {"category": {"$regex": f"^{category}$", "$options": "i"}},
            {"category_slug": category.lower()}
        ]

    if subcategory and subcategory.lower() != "all":
        subcat_filter = [
            {"subcategory": {"$regex": f"^{subcategory}$", "$options": "i"}},
            {"subcategory_slug": subcategory.lower().replace(" ", "-")}
        ]
        if "$or" in query:
            query = {"$and": [query, {"$or": subcat_filter}]}
        else:
            query["$or"] = subcat_filter

    if brand and brand.lower() != "all":
        query["brand"] = brand

    if min_price is not None or max_price is not None:
        price_query = {}
        if min_price is not None:
            price_query["$gte"] = min_price
        if max_price is not None:
            price_query["$lte"] = max_price
        query["discounted_price"] = price_query

    if min_rating is not None:
        query["rating"] = {"$gte": min_rating}

    if min_discount is not None:
        query["discount_percentage"] = {"$gte": min_discount}

    if in_stock_only:
        query["stock_quantity"] = {"$gt": 0}

    # Sorting
    sort_spec = []
    if sort_by in ("price_asc", "price-low-to-high"):
        sort_spec = [("discounted_price", 1)]
    elif sort_by in ("price_desc", "price-high-to-low"):
        sort_spec = [("discounted_price", -1)]
    elif sort_by in ("rating_desc", "rating", "customer_rating"):
        sort_spec = [("rating", -1), ("review_count", -1)]
    elif sort_by in ("popularity", "popular"):
        sort_spec = [("popularity_score", -1)]
    elif sort_by in ("discount_desc", "deals", "best_discount"):
        sort_spec = [("discount_percentage", -1)]
    elif sort_by in ("newest", "created_desc"):
        sort_spec = [("created_at", -1)]
    else:
        sort_spec = [("popularity_score", -1), ("product_id", 1)]

    cursor = products_collection.find(query)
    if sort_spec:
        cursor = cursor.sort(sort_spec)

    total = products_collection.count_documents(query)
    raw_products = list(cursor.skip(skip).limit(limit))
    products = [format_product_doc(p) for p in raw_products]

    total_pages = (total + limit - 1) // limit if limit > 0 else 1
    current_page = (skip // limit) + 1

    return {
        "total": total,
        "limit": limit,
        "skip": skip,
        "page": current_page,
        "total_pages": total_pages,
        "products": products
    }


@router.get("/categories")
def get_categories():
    """Returns all product categories with subcategories and counts"""
    categories = products_collection.distinct("category", {"is_active": True})
    categories.sort()

    category_details = []
    for cat in categories:
        count = products_collection.count_documents({"category": cat, "is_active": True})
        subcats = products_collection.distinct("subcategory", {"category": cat, "is_active": True})
        subcats.sort()
        category_details.append({
            "name": cat,
            "slug": cat.lower().replace(" & ", "-and-").replace(" ", "-"),
            "count": count,
            "subcategories": subcats
        })

    return {
        "categories": categories,
        "details": category_details
    }


@router.get("/subcategories")
def get_subcategories(category: Optional[str] = None):
    """Returns subcategories, optionally filtered by category"""
    query = {"is_active": True}
    if category and category.lower() != "all":
        query["$or"] = [
            {"category": {"$regex": f"^{category}$", "$options": "i"}},
            {"category_slug": category.lower()}
        ]
    subcats = products_collection.distinct("subcategory", query)
    subcats.sort()
    return {"subcategories": subcats}


@router.get("/brands")
def get_brands(category: Optional[str] = None, subcategory: Optional[str] = None):
    """Returns available brands for filtering"""
    query = {"is_active": True, "brand": {"$exists": True, "$ne": ""}}
    if category and category.lower() != "all":
        query["$or"] = [
            {"category": {"$regex": f"^{category}$", "$options": "i"}},
            {"category_slug": category.lower()}
        ]
    if subcategory and subcategory.lower() != "all":
        query["subcategory"] = {"$regex": f"^{subcategory}$", "$options": "i"}

    brands = products_collection.distinct("brand", query)
    brands = [b for b in brands if b and b.strip()]
    brands.sort()
    return {"brands": brands}


@router.get("/filters")
def get_filter_facets(category: Optional[str] = None):
    """
    Returns available filter ranges (brands, subcategories, price bounds)
    for dynamically building the PLP filter sidebar.
    """
    query = {"is_active": True}
    if category and category.lower() != "all":
        query["$or"] = [
            {"category": {"$regex": f"^{category}$", "$options": "i"}},
            {"category_slug": category.lower()}
        ]

    brands = products_collection.distinct("brand", query)
    brands = [b for b in brands if b]
    brands.sort()

    subcategories = products_collection.distinct("subcategory", query)
    subcategories.sort()

    # Price bounds
    min_doc = products_collection.find_one(query, sort=[("discounted_price", 1)])
    max_doc = products_collection.find_one(query, sort=[("discounted_price", -1)])
    min_price = min_doc.get("discounted_price", 0) if min_doc else 0
    max_price = max_doc.get("discounted_price", 100000) if max_doc else 100000

    return {
        "category": category,
        "brands": brands[:30],
        "subcategories": subcategories,
        "price_range": {
            "min": round(min_price, 2),
            "max": round(max_price, 2)
        },
        "rating_options": [4.5, 4.0, 3.5, 3.0],
        "discount_options": [50, 40, 30, 20, 10]
    }


@router.get("/trending")
def get_trending_products(limit: int = Query(12, ge=1, le=50)):
    """Trending products ranked by popularity score"""
    raw = list(
        products_collection.find({"is_active": True})
        .sort([("popularity_score", -1), ("rating", -1)])
        .limit(limit)
    )
    return {"products": [format_product_doc(p) for p in raw]}


@router.get("/bestsellers")
def get_bestsellers(limit: int = Query(12, ge=1, le=50)):
    """Best sellers ranked by review count and rating"""
    raw = list(
        products_collection.find({"is_active": True})
        .sort([("review_count", -1), ("rating", -1)])
        .limit(limit)
    )
    return {"products": [format_product_doc(p) for p in raw]}


@router.get("/top-rated")
def get_top_rated(limit: int = Query(12, ge=1, le=50)):
    """Top rated products with rating >= 4.5"""
    raw = list(
        products_collection.find({"is_active": True, "rating": {"$gte": 4.5}})
        .sort([("rating", -1), ("review_count", -1)])
        .limit(limit)
    )
    return {"products": [format_product_doc(p) for p in raw]}


@router.get("/deals")
def get_todays_deals(limit: int = Query(12, ge=1, le=50)):
    """Deals sorted by highest discount percentage"""
    raw = list(
        products_collection.find({"is_active": True, "discount_percentage": {"$gte": 30}})
        .sort([("discount_percentage", -1), ("popularity_score", -1)])
        .limit(limit)
    )
    return {"products": [format_product_doc(p) for p in raw]}


@router.get("/recently-added")
def get_recently_added(limit: int = Query(12, ge=1, le=50)):
    """Recently added products"""
    raw = list(
        products_collection.find({"is_active": True})
        .sort([("created_at", -1)])
        .limit(limit)
    )
    return {"products": [format_product_doc(p) for p in raw]}


@router.get("/search/suggestions")
def get_search_suggestions(q: str = Query(..., min_length=1)):
    """
    Autocomplete suggestions: returns top 6-8 matching product titles and brands
    """
    clean_q = q.strip()
    regex = {"$regex": f"^{clean_q}", "$options": "i"}
    contains_regex = {"$regex": clean_q, "$options": "i"}

    # Find matching brands
    matching_brands = products_collection.distinct("brand", {"brand": contains_regex, "is_active": True})[:4]

    # Find matching categories or subcategories
    matching_subcats = products_collection.distinct("subcategory", {"subcategory": contains_regex, "is_active": True})[:3]

    # Find matching titles
    matching_products = list(
        products_collection.find(
            {"title": contains_regex, "is_active": True},
            {"title": 1, "product_id": 1, "category": 1, "thumbnail": 1, "discounted_price": 1}
        ).limit(6)
    )

    items = []
    for p in matching_products:
        items.append({
            "product_id": p.get("product_id"),
            "title": p.get("title"),
            "category": p.get("category"),
            "thumbnail": p.get("thumbnail"),
            "price": p.get("discounted_price")
        })

    return {
        "query": clean_q,
        "brands": matching_brands,
        "subcategories": matching_subcats,
        "products": items
    }


@router.get("/search")
def search_products(
    q: str = Query(..., min_length=1),
    category: Optional[str] = None,
    sort_by: Optional[str] = Query("default"),
    limit: int = Query(24, ge=1, le=50),
    skip: int = Query(0, ge=0),
    page: Optional[int] = Query(None, ge=1)
):
    """
    Comprehensive product search supporting full-text search and partial regex matching.
    """
    if page is not None:
        skip = (page - 1) * limit

    clean_q = q.strip()
    regex = {"$regex": clean_q, "$options": "i"}

    # First attempt MongoDB Text Search
    text_query = {
        "is_active": True,
        "$text": {"$search": clean_q}
    }
    if category and category.lower() != "all":
        text_query["category"] = {"$regex": f"^{category}$", "$options": "i"}

    total = products_collection.count_documents(text_query)
    raw_products = []

    if total > 0:
        cursor = products_collection.find(
            text_query,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})])
        raw_products = list(cursor.skip(skip).limit(limit))
    else:
        # Fallback to regex partial matching on title, brand, category, tags
        regex_query = {
            "is_active": True,
            "$or": [
                {"title": regex},
                {"name": regex},
                {"brand": regex},
                {"category": regex},
                {"subcategory": regex},
                {"tags": regex}
            ]
        }
        if category and category.lower() != "all":
            regex_query["category"] = {"$regex": f"^{category}$", "$options": "i"}

        total = products_collection.count_documents(regex_query)
        cursor = products_collection.find(regex_query).sort([("popularity_score", -1)])
        raw_products = list(cursor.skip(skip).limit(limit))

    products = [format_product_doc(p) for p in raw_products]
    total_pages = (total + limit - 1) // limit if limit > 0 else 1

    return {
        "query": clean_q,
        "total": total,
        "limit": limit,
        "skip": skip,
        "page": (skip // limit) + 1,
        "total_pages": total_pages,
        "products": products
    }


@router.get("/{product_id}")
def get_product_by_id(product_id: str):
    """Fetch product details by ID or slug"""
    query = None
    if product_id.isdigit():
        p_num = int(product_id)
        query = {
            "$or": [
                {"product_id": p_num},
                {"external_id": p_num}
            ],
            "is_active": True
        }
    else:
        try:
            query = {"_id": ObjectId(product_id), "is_active": True}
        except Exception:
            query = {
                "$or": [
                    {"product_id": product_id},
                    {"external_id": product_id},
                    {"slug": product_id}
                ],
                "is_active": True
            }

    product = products_collection.find_one(query)
    if not product and product_id.isdigit():
        p_num = int(product_id)
        product = products_collection.find_one({"$or": [{"product_id": p_num}, {"external_id": p_num}]})

    if not product:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found")

    return format_product_doc(product)


@router.get("/{product_id}/similar")
def get_similar_products(product_id: str, limit: int = Query(6, ge=1, le=12)):
    """Fetch similar products in same subcategory / category"""
    p_num = int(product_id) if product_id.isdigit() else None
    query = {"$or": [{"product_id": p_num}, {"external_id": p_num}]} if p_num else {"product_id": product_id}
    base_prod = products_collection.find_one(query)

    if not base_prod:
        return {"products": []}

    subcat = base_prod.get("subcategory")
    cat = base_prod.get("category")
    active_id = base_prod.get("product_id") or base_prod.get("external_id")

    # Match same subcategory first, then same category
    sim_query = {
        "is_active": True,
        "product_id": {"$ne": active_id},
        "external_id": {"$ne": active_id}
    }
    if subcat:
        sim_query["subcategory"] = subcat
    elif cat:
        sim_query["category"] = cat

    raw_items = list(products_collection.find(sim_query).sort([("rating", -1), ("popularity_score", -1)]).limit(limit))
    return {"products": [format_product_doc(p) for p in raw_items]}


@router.get("/{product_id}/frequently-bought-together")
def get_frequently_bought_together(product_id: str, limit: int = Query(3, ge=1, le=6)):
    """Complementary products from same category or complementary accessories"""
    p_num = int(product_id) if product_id.isdigit() else None
    query = {"$or": [{"product_id": p_num}, {"external_id": p_num}]} if p_num else {"product_id": product_id}
    base_prod = products_collection.find_one(query)

    if not base_prod:
        return {"products": []}

    cat = base_prod.get("category")
    active_id = base_prod.get("product_id") or base_prod.get("external_id")

    fbt_query = {
        "is_active": True,
        "category": cat,
        "product_id": {"$ne": active_id},
        "external_id": {"$ne": active_id}
    }

    raw_items = list(products_collection.find(fbt_query).sort([("popularity_score", -1)]).limit(limit))
    return {"products": [format_product_doc(p) for p in raw_items]}

"""
SmartCart AI — DummyJSON Catalog Importer & Normalizer (Phase 8)
Features:
- Offline reproducible loading from data/raw/dummyjson/ (or --refresh)
- Strict department validation with data/mappings/dummyjson_category_map.json
- USD to INR conversion with configurable exchange rate (default 85)
- MRP derivation from discountPercentage
- Fallback for unbranded items ('Unbranded' + flagged)
- Product reviews extraction into product_reviews table
- Idempotent upserts on (source, source_id)
- Sets is_active=False on seeded non-book items; keeps seeded Books active
- Supports --dry-run, --refresh, --limit, --usd-rate
"""

import os
import sys
import json
import argparse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.database import SessionLocal, init_db
from app.models import Product, ProductReview
from app.config import settings


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dummyjson"
MAPPING_FILE = PROJECT_ROOT / "data" / "mappings" / "dummyjson_category_map.json"


def download_raw_data() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading products from DummyJSON (limit=0)...")
    req = urllib.request.Request("https://dummyjson.com/products?limit=0", headers={"User-Agent": "SmartCart-AI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        products_data = json.loads(resp.read().decode("utf-8"))
    with open(RAW_DATA_DIR / "products.json", "w", encoding="utf-8") as f:
        json.dump(products_data, f, indent=2)

    print("Downloading categories from DummyJSON...")
    req = urllib.request.Request("https://dummyjson.com/products/categories", headers={"User-Agent": "SmartCart-AI/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        categories_data = json.loads(resp.read().decode("utf-8"))
    with open(RAW_DATA_DIR / "categories.json", "w", encoding="utf-8") as f:
        json.dump(categories_data, f, indent=2)

    print(f"Downloaded {len(products_data.get('products', []))} products and {len(categories_data)} categories.")


def load_category_map() -> Dict[str, Dict[str, str]]:
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Category mapping file not found at {MAPPING_FILE}")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def import_dummyjson_catalog(
    dry_run: bool = False,
    refresh: bool = False,
    limit: Optional[int] = None,
    usd_rate: Optional[float] = None
) -> Dict[str, Any]:
    if refresh or not (RAW_DATA_DIR / "products.json").exists():
        download_raw_data()

    with open(RAW_DATA_DIR / "products.json", "r", encoding="utf-8") as f:
        raw_payload = json.load(f)
    raw_products = raw_payload.get("products", [])

    category_map = load_category_map()
    rate = usd_rate or settings.USD_TO_INR_RATE or 85.0

    print(f"Starting DummyJSON import (dry_run={dry_run}, USD_RATE={rate}, limit={limit})...")
    print(f"Found {len(raw_products)} raw products in offline store.")

    # 1. Verification of category mappings
    for p in raw_products:
        cat = p.get("category")
        if not cat or cat not in category_map:
            raise ValueError(f"FATAL: Unmapped DummyJSON category '{cat}' for product '{p.get('title')}'. Import failed.")

    if limit and limit > 0:
        raw_products = raw_products[:limit]

    stats = {
        "total_evaluated": len(raw_products),
        "imported_active": 0,
        "excluded_skipped": 0,
        "unbranded_count": 0,
        "reviews_count": 0,
        "seeded_deactivated": 0,
        "seeded_books_preserved": 0
    }

    db = SessionLocal()
    try:
        now_utc = datetime.now(timezone.utc)

        # 2. Update seeded products: deactivate non-books, ensure books stay active
        if not dry_run:
            seeded_non_books = (
                db.query(Product)
                .filter(Product.source == "seed", Product.category != "Books")
                .all()
            )
            for sp in seeded_non_books:
                sp.is_active = False
            stats["seeded_deactivated"] = len(seeded_non_books)

            seeded_books = (
                db.query(Product)
                .filter(Product.source == "seed", Product.category == "Books")
                .all()
            )
            for sb in seeded_books:
                sb.is_active = True
            stats["seeded_books_preserved"] = len(seeded_books)

        # 3. Process DummyJSON products
        for p in raw_products:
            cat_slug = p["category"]
            mapping_info = category_map[cat_slug]
            dept = mapping_info["department"]

            if dept.lower() == "excluded":
                stats["excluded_skipped"] += 1
                continue

            # Price calculations
            raw_usd = float(p.get("price", 0.0))
            price_inr = round(raw_usd * rate, 2)
            discount_pct = float(p.get("discountPercentage", 0.0))
            mrp_inr = round(price_inr / max(0.01, (1.0 - (discount_pct / 100.0))), 2)
            if mrp_inr < price_inr:
                mrp_inr = price_inr

            # Brand handling
            brand = p.get("brand")
            is_unbranded = False
            if not brand or not str(brand).strip():
                brand = "Unbranded"
                is_unbranded = True
                stats["unbranded_count"] += 1
            else:
                brand = str(brand).strip()

            # Placeholder popularity derived from rating and discount
            rating = float(p.get("rating", 4.0))
            pop_score = round(min(100.0, max(10.0, (rating / 5.0) * 60.0 + (min(50.0, discount_pct) / 50.0) * 40.0)), 1)

            # Images
            images = p.get("images", [])
            thumbnail = p.get("thumbnail") or (images[0] if images else "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=600")

            # Attributes dict
            attrs = {
                "sku": p.get("sku"),
                "weight": p.get("weight"),
                "dimensions": p.get("dimensions"),
                "warrantyInformation": p.get("warrantyInformation"),
                "shippingInformation": p.get("shippingInformation"),
                "availabilityStatus": p.get("availabilityStatus", "In Stock"),
                "returnPolicy": p.get("returnPolicy"),
                "tags": p.get("tags", []),
                "is_unbranded": is_unbranded,
                "dummyjson_category": cat_slug,
                "gallery": images
            }

            source_id_str = str(p["id"])

            if not dry_run:
                # Idempotent upsert by (source, source_id)
                product_obj = (
                    db.query(Product)
                    .filter(Product.source == "dummyjson", Product.source_id == source_id_str)
                    .first()
                )

                if not product_obj:
                    product_obj = Product(
                        name=p.get("title", f"Product #{source_id_str}"),
                        category=dept,
                        brand=brand,
                        price=price_inr,
                        mrp=mrp_inr,
                        discount_pct=discount_pct,
                        rating=rating,
                        rating_count=len(p.get("reviews", [])) or 15,
                        popularity_score=pop_score,
                        description=p.get("description", ""),
                        image_url=thumbnail,
                        attributes=attrs,
                        stock=int(p.get("stock", 50)),
                        source="dummyjson",
                        source_id=source_id_str,
                        is_active=True,
                        raw_price_usd=raw_usd,
                        imported_at=now_utc
                    )
                    db.add(product_obj)
                    db.flush()
                else:
                    # Update fields idempotently
                    product_obj.name = p.get("title", product_obj.name)
                    product_obj.category = dept
                    product_obj.brand = brand
                    product_obj.price = price_inr
                    product_obj.mrp = mrp_inr
                    product_obj.discount_pct = discount_pct
                    product_obj.rating = rating
                    product_obj.rating_count = len(p.get("reviews", [])) or product_obj.rating_count
                    product_obj.popularity_score = pop_score
                    product_obj.description = p.get("description", product_obj.description)
                    product_obj.image_url = thumbnail
                    product_obj.attributes = attrs
                    product_obj.stock = int(p.get("stock", product_obj.stock))
                    product_obj.is_active = True
                    product_obj.raw_price_usd = raw_usd
                    product_obj.imported_at = now_utc

                # Reviews extraction into product_reviews
                raw_reviews = p.get("reviews", [])
                for rev in raw_reviews:
                    rev_name = rev.get("reviewerName") or "Verified Customer"
                    rev_comment = rev.get("comment") or "Great purchase."
                    rev_rating = float(rev.get("rating", 5.0))
                    rev_date_str = rev.get("date")
                    rev_date = None
                    if rev_date_str:
                        try:
                            rev_date = datetime.fromisoformat(rev_date_str.replace("Z", "+00:00"))
                        except Exception:
                            pass

                    # Check if review already exists for this product
                    existing_rev = (
                        db.query(ProductReview)
                        .filter(
                            ProductReview.product_id == product_obj.id,
                            ProductReview.reviewer_name == rev_name,
                            ProductReview.comment == rev_comment
                        )
                        .first()
                    )
                    if not existing_rev:
                        new_rev = ProductReview(
                            product_id=product_obj.id,
                            reviewer_name=rev_name,
                            rating=rev_rating,
                            comment=rev_comment,
                            date=rev_date,
                            source="dummyjson",
                            created_at=now_utc
                        )
                        db.add(new_rev)
                        stats["reviews_count"] += 1

            stats["imported_active"] += 1

        if not dry_run:
            db.commit()
            print("Database transaction successfully committed.")

    finally:
        db.close()

    print("\n--- IMPORT SUMMARY ---")
    print(f"Total Evaluated: {stats['total_evaluated']}")
    print(f"Imported Active: {stats['imported_active']}")
    print(f"Excluded (Skipped): {stats['excluded_skipped']}")
    print(f"Unbranded Items: {stats['unbranded_count']}")
    print(f"Product Reviews Created: {stats['reviews_count']}")
    print(f"Seeded Products Deactivated: {stats['seeded_deactivated']}")
    print(f"Seeded Books Preserved: {stats['seeded_books_preserved']}")
    print("-----------------------\n")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Import DummyJSON Catalog to SmartCart AI")
    parser.add_argument("--dry-run", action="store_true", help="Perform mapping and validation without writing to DB")
    parser.add_argument("--refresh", action="store_true", help="Re-download raw JSON files from dummyjson.com")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items imported")
    parser.add_argument("--usd-rate", type=float, default=None, help="Override USD to INR exchange rate")
    args = parser.parse_args()

    import_dummyjson_catalog(
        dry_run=args.dry_run,
        refresh=args.refresh,
        limit=args.limit,
        usd_rate=args.usd_rate
    )


if __name__ == "__main__":
    main()

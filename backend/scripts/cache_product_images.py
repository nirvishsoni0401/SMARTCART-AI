"""
SmartCart AI — Offline Image Cache Script
Downloads thumbnails and gallery images from external CDNs (DummyJSON)
into backend/static/products/ for offline-safe local serving.
"""

import os
import sys
import json
import time
import sqlite3
import argparse
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_DIR / "backend" / "smartcart.db"
STATIC_PRODUCTS_DIR = ROOT_DIR / "backend" / "static" / "products"

USER_AGENT = "SmartCart-ImageCacher/1.0 (+http://smartcart.ai)"


def download_image(url: str, dest_path: Path, timeout: float = 6.0, retries: int = 2) -> bool:
    """Downloads a single image to dest_path idempotently."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 200:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = dest_path.with_suffix(".tmp")
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=16384):
                        if chunk:
                            f.write(chunk)
                temp_path.replace(dest_path)
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def cache_product(product: dict, force: bool = False, dry_run: bool = False) -> dict:
    """Caches thumbnail and gallery images for a single product."""
    pid = product["id"]
    img_url = product["image_url"]
    gallery = product.get("gallery", [])

    result = {
        "id": pid,
        "name": product["name"],
        "thumb_cached": False,
        "local_thumb_url": None,
        "gallery_cached_count": 0,
        "local_gallery": [],
        "errors": []
    }

    if not img_url or not img_url.startswith("http"):
        return result

    # Determine file extension from URL
    ext = Path(urlparse(img_url).path).suffix or ".webp"
    if ext.lower() not in [".webp", ".png", ".jpg", ".jpeg", ".svg"]:
        ext = ".webp"

    thumb_filename = f"prod_{pid}_thumb{ext}"
    thumb_path = STATIC_PRODUCTS_DIR / thumb_filename

    if dry_run:
        result["thumb_cached"] = thumb_path.exists()
        result["local_thumb_url"] = f"/static/products/{thumb_filename}"
        return result

    # Download thumbnail
    if not thumb_path.exists() or force:
        ok = download_image(img_url, thumb_path)
        if ok:
            result["thumb_cached"] = True
            result["local_thumb_url"] = f"/static/products/{thumb_filename}"
        else:
            result["errors"].append(f"Failed to download thumbnail: {img_url}")
    else:
        result["thumb_cached"] = True
        result["local_thumb_url"] = f"/static/products/{thumb_filename}"

    # Download gallery images
    local_gallery = []
    for idx, g_url in enumerate(gallery):
        if not g_url or not g_url.startswith("http"):
            continue
        g_ext = Path(urlparse(g_url).path).suffix or ".webp"
        g_filename = f"prod_{pid}_g{idx}{g_ext}"
        g_path = STATIC_PRODUCTS_DIR / g_filename

        if not g_path.exists() or force:
            ok = download_image(g_url, g_path)
            if ok:
                result["gallery_cached_count"] += 1
                local_gallery.append(f"/static/products/{g_filename}")
        else:
            result["gallery_cached_count"] += 1
            local_gallery.append(f"/static/products/{g_filename}")

    result["local_gallery"] = local_gallery
    return result


def main():
    parser = argparse.ArgumentParser(description="Cache remote product images locally")
    parser.add_argument("--dry-run", action="store_true", help="Report planned downloads without saving")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products to process")
    parser.add_argument("--force", action="store_true", help="Re-download even if already cached")
    parser.add_argument("--rewrite-image-url", action="store_true", help="Rewrite primary image_url to local path")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent download workers")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    STATIC_PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, category, brand, source, image_url, attributes
        FROM products
        WHERE is_active = 1 AND image_url LIKE 'http%'
        ORDER BY id ASC
    """)
    rows = c.fetchall()

    if args.limit:
        rows = rows[:args.limit]

    print(f"Found {len(rows)} active products with remote image URLs.")
    if args.dry_run:
        print("[DRY-RUN MODE] No files will be downloaded or database modified.")

    products = []
    for r in rows:
        pid, name, cat, brand, src, img_url, attr_json = r
        gallery = []
        if attr_json:
            try:
                attrs = json.loads(attr_json)
                if isinstance(attrs, dict) and "gallery" in attrs:
                    gallery = attrs["gallery"]
            except Exception:
                pass
        products.append({
            "id": pid,
            "name": name,
            "category": cat,
            "brand": brand,
            "source": src,
            "image_url": img_url,
            "gallery": gallery,
            "raw_attributes": attr_json
        })

    cached_count = 0
    updated_db_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(cache_product, p, args.force, args.dry_run): p
            for p in products
        }
        for future in as_completed(future_map):
            p = future_map[future]
            res = future.result()
            if res["thumb_cached"]:
                cached_count += 1

                if not args.dry_run:
                    # Update database attributes with local_image_url
                    try:
                        attrs = json.loads(p["raw_attributes"]) if p["raw_attributes"] else {}
                    except Exception:
                        attrs = {}
                    
                    attrs["local_image_url"] = res["local_thumb_url"]
                    attrs["cdn_image_url"] = p["image_url"]
                    if res["local_gallery"]:
                        attrs["local_gallery"] = res["local_gallery"]

                    new_image_url = res["local_thumb_url"] if args.rewrite_image_url else p["image_url"]
                    c.execute("""
                        UPDATE products
                        SET attributes = ?, image_url = ?
                        WHERE id = ?
                    """, (json.dumps(attrs), new_image_url, p["id"]))
                    updated_db_count += 1

    if not args.dry_run and updated_db_count > 0:
        conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("SMARTCART AI — IMAGE CACHING SUMMARY")
    print("=" * 60)
    print(f"Products Processed:  {len(products)}")
    print(f"Successfully Cached: {cached_count} / {len(products)}")
    print(f"Database Rows Updated: {updated_db_count}")
    print(f"Target Directory:    {STATIC_PRODUCTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

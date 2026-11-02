import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.main import app
from app.database import products_collection

client = TestClient(app)

def run_api_audit():
    print("===================================================================")
    print("                 2. API ENDPOINTS AUDIT REPORT                     ")
    print("===================================================================")
    
    results = {}
    
    # Check 1: Live DB vs Hardcoded Check
    # Insert a unique canary doc into DB, query via API, then remove
    canary_id = 999999
    products_collection.delete_one({"product_id": canary_id})
    products_collection.insert_one({
        "product_id": canary_id,
        "external_id": canary_id,
        "title": "AUDIT_CANARY_UNIQUE_ITEM_XYZ",
        "name": "AUDIT_CANARY_UNIQUE_ITEM_XYZ",
        "description": "Canary product for dynamic DB verification",
        "brand": "CanaryBrand",
        "category": "Electronics",
        "category_slug": "electronics",
        "subcategory": "Smartphones",
        "price": 99999.0,
        "discounted_price": 49999.0,
        "discount_percentage": 50,
        "rating": 4.9,
        "review_count": 100,
        "stock_quantity": 25,
        "stock": 25,
        "availability": "In Stock",
        "images": ["https://images.unsplash.com/test.jpg"],
        "thumbnail": "https://images.unsplash.com/test.jpg",
        "seller": "Canary Retail",
        "popularity_score": 95.0,
        "is_active": True,
        "created_at": "2026-11-02T00:00:00Z",
        "updated_at": "2026-11-02T00:00:00Z"
    })
    
    canary_res = client.get(f"/api/products/{canary_id}")
    is_dynamic = canary_res.status_code == 200 and canary_res.json().get("title") == "AUDIT_CANARY_UNIQUE_ITEM_XYZ"
    products_collection.delete_one({"product_id": canary_id})
    print(f"1. Actual DB records vs Hardcoded Responses: {'VERIFIED DYNAMIC (Real MongoDB query)' if is_dynamic else 'FAILED (Hardcoded or missing)'}")
    results["dynamic_db"] = is_dynamic

    # Check 2: Pagination Accuracy
    page1 = client.get("/api/products/?limit=10&page=1").json()
    page2 = client.get("/api/products/?limit=10&page=2").json()
    p1_ids = [p["product_id"] for p in page1["products"]]
    p2_ids = [p["product_id"] for p in page2["products"]]
    pagination_ok = len(p1_ids) == 10 and len(p2_ids) == 10 and len(set(p1_ids).intersection(set(p2_ids))) == 0
    print(f"2. Pagination Accuracy (Page 1 vs 2 non-overlapping): {'VERIFIED' if pagination_ok else 'FAILED'}")
    results["pagination"] = pagination_ok

    # Check 3: Category Filtering
    cat_res = client.get("/api/products/?category=Electronics&limit=15").json()
    cat_matches = all(p["category"].lower() == "electronics" for p in cat_res["products"])
    print(f"3. Category Filtering (Electronics): {'VERIFIED' if cat_matches and len(cat_res['products']) == 15 else 'FAILED'}")
    results["category_filter"] = cat_matches

    # Check 4: Subcategory Filtering
    sub_res = client.get("/api/products/?subcategory=Smartphones&limit=15").json()
    sub_matches = all(p.get("subcategory", "").lower() == "smartphones" for p in sub_res["products"])
    print(f"4. Subcategory Filtering (Smartphones): {'VERIFIED' if sub_matches and len(sub_res['products']) == 15 else 'FAILED'}")
    results["subcategory_filter"] = sub_matches

    # Check 5: Brand Filtering
    brand_res = client.get("/api/products/?brand=Apple&limit=15").json()
    brand_matches = all(p.get("brand") == "Apple" for p in brand_res["products"])
    print(f"5. Brand Filtering (Apple): {'VERIFIED' if brand_matches and len(brand_res['products']) > 0 else 'FAILED'}")
    results["brand_filter"] = brand_matches

    # Check 6: Price Range Filtering
    price_res = client.get("/api/products/?min_price=10000&max_price=20000&limit=20").json()
    price_matches = all(10000 <= p["discounted_price"] <= 20000 for p in price_res["products"])
    print(f"6. Price Range Filtering (Rs. 10,000 - Rs. 20,000): {'VERIFIED' if price_matches and len(price_res['products']) > 0 else 'FAILED'}")
    results["price_filter"] = price_matches

    # Check 7: Rating Filtering
    rating_res = client.get("/api/products/?min_rating=4.5&limit=20").json()
    rating_matches = all(p["rating"] >= 4.5 for p in rating_res["products"])
    print(f"7. Rating Filtering (>= 4.5 stars): {'VERIFIED' if rating_matches and len(rating_res['products']) > 0 else 'FAILED'}")
    results["rating_filter"] = rating_matches

    # Check 8: Discount Filtering
    disc_res = client.get("/api/products/?min_discount=50&limit=20").json()
    disc_matches = all(p.get("discount_percentage", 0) >= 50 for p in disc_res["products"])
    print(f"8. Discount Filtering (>= 50% OFF): {'VERIFIED' if disc_matches and len(disc_res['products']) > 0 else 'FAILED'}")
    results["discount_filter"] = disc_matches

    # Check 9: In-Stock Filtering
    stock_res = client.get("/api/products/?in_stock_only=true&limit=20").json()
    stock_matches = all(p.get("stock_quantity", 0) > 0 for p in stock_res["products"])
    print(f"9. Stock Filtering (In Stock Only): {'VERIFIED' if stock_matches and len(stock_res['products']) > 0 else 'FAILED'}")
    results["stock_filter"] = stock_matches

    # Check 10: Sorting Accuracy
    # Price ASC
    sort_p_asc = client.get("/api/products/?sort_by=price_asc&limit=30").json()["products"]
    p_asc_prices = [p["discounted_price"] for p in sort_p_asc]
    p_asc_ok = all(p_asc_prices[i] <= p_asc_prices[i+1] for i in range(len(p_asc_prices)-1))
    
    # Price DESC
    sort_p_desc = client.get("/api/products/?sort_by=price_desc&limit=30").json()["products"]
    p_desc_prices = [p["discounted_price"] for p in sort_p_desc]
    p_desc_ok = all(p_desc_prices[i] >= p_desc_prices[i+1] for i in range(len(p_desc_prices)-1))

    # Rating DESC
    sort_r_desc = client.get("/api/products/?sort_by=rating_desc&limit=30").json()["products"]
    r_desc_ratings = [p["rating"] for p in sort_r_desc]
    r_desc_ok = all(r_desc_ratings[i] >= r_desc_ratings[i+1] for i in range(len(r_desc_ratings)-1))

    # Discount DESC
    sort_d_desc = client.get("/api/products/?sort_by=discount_desc&limit=30").json()["products"]
    d_desc_discs = [p.get("discount_percentage", 0) for p in sort_d_desc]
    d_desc_ok = all(d_desc_discs[i] >= d_desc_discs[i+1] for i in range(len(d_desc_discs)-1))

    print(f"10. Sorting Accuracy:")
    print(f"    - price_asc:     {'VERIFIED' if p_asc_ok else 'FAILED'} (min: Rs. {p_asc_prices[0]}, max: Rs. {p_asc_prices[-1]})")
    print(f"    - price_desc:    {'VERIFIED' if p_desc_ok else 'FAILED'} (max: Rs. {p_desc_prices[0]}, min: Rs. {p_desc_prices[-1]})")
    print(f"    - rating_desc:   {'VERIFIED' if r_desc_ok else 'FAILED'} (max: {r_desc_ratings[0]}, min: {r_desc_ratings[-1]})")
    print(f"    - discount_desc: {'VERIFIED' if d_desc_ok else 'FAILED'} (max: {d_desc_discs[0]}%, min: {d_desc_discs[-1]}%)")
    results["sorting"] = p_asc_ok and p_desc_ok and r_desc_ok and d_desc_ok

    # Check 11: Search Relevance
    search_res = client.get("/api/products/search?q=laptop&limit=15").json()
    search_count = search_res.get("total", 0)
    search_relevance = any("laptop" in (p["title"] + " " + p["description"] + " " + p.get("subcategory","")).lower() for p in search_res["products"])
    print(f"11. Search Relevance (q='laptop'): {'VERIFIED' if search_relevance and search_count > 0 else 'FAILED'} ({search_count} matches)")
    results["search"] = search_relevance

    # Check 12: Autocomplete Suggestions
    sug_res = client.get("/api/products/search/suggestions?q=smart").json()
    has_brands_or_prods = len(sug_res.get("products", [])) > 0 or len(sug_res.get("subcategories", [])) > 0
    print(f"12. Autocomplete Suggestions (q='smart'): {'VERIFIED' if has_brands_or_prods else 'FAILED'} ({len(sug_res.get('products', []))} products suggested)")
    results["autocomplete"] = has_brands_or_prods

    # Check 13: Similar Products
    sample_p = products_collection.find_one({"category": "Electronics"})
    pid = sample_p["product_id"]
    sim_res = client.get(f"/api/products/{pid}/similar?limit=4").json()
    sim_prods = sim_res.get("products", [])
    sim_ok = len(sim_prods) == 4 and all(p["category"] == sample_p["category"] and p["product_id"] != pid for p in sim_prods)
    print(f"13. Similar Products for ID {pid}: {'VERIFIED' if sim_ok else 'FAILED'} ({len(sim_prods)} returned)")
    results["similar"] = sim_ok

    # Check 14: Frequently Bought Together
    fbt_res = client.get(f"/api/products/{pid}/frequently-bought-together?limit=3").json()
    fbt_prods = fbt_res.get("products", [])
    fbt_ok = len(fbt_prods) == 3 and all(p["product_id"] != pid for p in fbt_prods)
    print(f"14. Frequently Bought Together for ID {pid}: {'VERIFIED' if fbt_ok else 'FAILED'} ({len(fbt_prods)} returned)")
    results["fbt"] = fbt_ok

    print("\nSummary of API Audits:", results)

if __name__ == "__main__":
    run_api_audit()

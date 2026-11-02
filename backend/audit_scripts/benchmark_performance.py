import time
import statistics
import sys
from pathlib import Path

backend_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_root))

from app.database import db, products_collection
from fastapi.testclient import TestClient
from app.main import app

def benchmark_db_queries():
    print("===================================================================")
    print("                6. PERFORMANCE BENCHMARK AUDIT                     ")
    print("===================================================================")
    print("\n--- A. MongoDB Index & Query Execution Profile ---")
    
    # 1. Category Filter Query with Sorting
    explain_cat = db.command("explain", {
        "find": "products",
        "filter": {"category": "Electronics", "is_active": True},
        "sort": {"popularity_score": -1},
        "limit": 20
    }, verbosity="executionStats")
    
    stats_cat = explain_cat.get("executionStats", {})
    print(f"Query 1: category='Electronics' + sort(popularity_score=-1) limit 20")
    print(f"  Execution Time      : {stats_cat.get('executionTimeMillis')} ms")
    print(f"  Docs Returned       : {stats_cat.get('nReturned')}")
    print(f"  Total Docs Examined : {stats_cat.get('totalDocsExamined')}")
    print(f"  Total Keys Examined : {stats_cat.get('totalKeysExamined')}")
    
    # 2. Text Search Query
    explain_search = db.command("explain", {
        "find": "products",
        "filter": {"$text": {"$search": "laptop"}, "is_active": True},
        "limit": 20
    }, verbosity="executionStats")
    
    stats_search = explain_search.get("executionStats", {})
    print(f"\nQuery 2: $text search 'laptop' limit 20")
    print(f"  Execution Time      : {stats_search.get('executionTimeMillis')} ms")
    print(f"  Docs Returned       : {stats_search.get('nReturned')}")
    print(f"  Total Docs Examined : {stats_search.get('totalDocsExamined')}")
    print(f"  Total Keys Examined : {stats_search.get('totalKeysExamined')}")

    # 3. Product Lookup by product_id index
    first_p = products_collection.find_one({"is_active": True})
    p_id = first_p.get("product_id") if first_p else 1
    
    explain_id = db.command("explain", {
        "find": "products",
        "filter": {"product_id": p_id}
    }, verbosity="executionStats")
    
    stats_id = explain_id.get("executionStats", {})
    print(f"\nQuery 3: Single item lookup filter {{product_id: {p_id}}}")
    print(f"  Execution Time      : {stats_id.get('executionTimeMillis')} ms")
    print(f"  Docs Returned       : {stats_id.get('nReturned')}")
    print(f"  Total Docs Examined : {stats_id.get('totalDocsExamined')}")
    print(f"  Total Keys Examined : {stats_id.get('totalKeysExamined')}")

def benchmark_api_endpoints():
    print("\n--- B. API Endpoint Response Time Benchmarks (10 warm iterations each) ---")
    
    first_p = products_collection.find_one({"is_active": True})
    p_id = first_p.get("product_id") if first_p else 1
    
    endpoints = [
        ("Product Listing (/api/products/?limit=20)", "/api/products/?limit=20"),
        ("Category Filter (/api/products/?category=Electronics&limit=20)", "/api/products/?category=Electronics&limit=20"),
        ("Subcategory Filter (/api/products/?subcategory=Smartphones&limit=20)", "/api/products/?subcategory=Smartphones&limit=20"),
        ("Brand Filter (/api/products/?brand=Apple&limit=20)", "/api/products/?brand=Apple&limit=20"),
        ("Multi-filter (cat+brand+price+rating)", "/api/products/?category=Electronics&brand=Apple&min_price=10000&max_price=200000&min_rating=4.0"),
        ("Text Search (/api/products/search?q=laptop)", "/api/products/search?q=laptop"),
        ("Search Autocomplete (/api/products/search/suggestions?q=sam)", "/api/products/search/suggestions?q=sam"),
        ("Category Facets (/api/products/filters?category=Electronics)", "/api/products/filters?category=Electronics"),
        ("Single Product Detail (/api/products/" + str(p_id) + ")", f"/api/products/{p_id}"),
        ("Similar Products (/api/products/" + str(p_id) + "/similar)", f"/api/products/{p_id}/similar"),
        ("Frequently Bought (/api/products/" + str(p_id) + "/frequently-bought-together)", f"/api/products/{p_id}/frequently-bought-together"),
        ("Personalized Recs (/api/recommendations/?user_id=bench_user)", "/api/recommendations/?user_id=bench_user"),
        ("Trending Products (/api/products/trending?limit=12)", "/api/products/trending?limit=12"),
        ("Deals (/api/products/deals?limit=12)", "/api/products/deals?limit=12"),
    ]
    
    with TestClient(app) as client:
        # Warmup
        client.get("/api/products/?limit=5")
        
        for name, path in endpoints:
            latencies = []
            status = 200
            for _ in range(10):
                t0 = time.perf_counter()
                res = client.get(path)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)
                status = res.status_code
            
            avg_ms = statistics.mean(latencies)
            min_ms = min(latencies)
            max_ms = max(latencies)
            median_ms = statistics.median(latencies)
            print(f"  {name:<65} | HTTP {status} | Avg: {avg_ms:6.2f} ms | Med: {median_ms:6.2f} ms | Min: {min_ms:5.2f} ms | Max: {max_ms:6.2f} ms")

if __name__ == "__main__":
    benchmark_db_queries()
    benchmark_api_endpoints()

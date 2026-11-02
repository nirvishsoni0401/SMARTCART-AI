import sys
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend folder is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.main import app

client = TestClient(app)


def run_tests():
    print("--- 1. Testing Root & Docs ---")
    res = client.get("/")
    assert res.status_code == 200, f"Root failed: {res.text}"
    print("  Root: OK")

    print("--- 2. Testing Products & Categories ---")
    res = client.get("/api/products/?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "products" in data
    assert len(data["products"]) == 5
    print(f"  Products listing: OK (Total: {data['total']:,})")

    res = client.get("/api/products/categories")
    assert res.status_code == 200
    cats = res.json()["categories"]
    assert len(cats) >= 10
    print(f"  Categories: OK ({len(cats)} categories found)")

    first_prod_id = data["products"][0]["product_id"]
    res = client.get(f"/api/products/{first_prod_id}")
    assert res.status_code == 200
    assert res.json()["title"] == data["products"][0]["title"]
    print(f"  Product details for ID {first_prod_id}: OK")

    print("--- 3. Testing New Amazon/Flipkart Section Endpoints ---")
    trend_res = client.get("/api/products/trending?limit=4")
    assert trend_res.status_code == 200
    assert len(trend_res.json()["products"]) == 4
    print("  Trending products: OK")

    best_res = client.get("/api/products/bestsellers?limit=4")
    assert best_res.status_code == 200
    assert len(best_res.json()["products"]) == 4
    print("  Bestsellers: OK")

    deals_res = client.get("/api/products/deals?limit=4")
    assert deals_res.status_code == 200
    print("  Deals: OK")

    top_res = client.get("/api/products/top-rated?limit=4")
    assert top_res.status_code == 200
    print("  Top rated: OK")

    filters_res = client.get("/api/products/filters?category=Electronics")
    assert filters_res.status_code == 200
    assert "brands" in filters_res.json()
    print("  Filters metadata: OK")

    sug_res = client.get("/api/products/search/suggestions?q=phone")
    assert sug_res.status_code == 200
    assert "products" in sug_res.json()
    print("  Search suggestions: OK")

    print("--- 4. Testing Similar & Frequently Bought Together ---")
    sim_res = client.get(f"/api/products/{first_prod_id}/similar?limit=3")
    assert sim_res.status_code == 200
    print("  Similar products: OK")

    fbt_res = client.get(f"/api/products/{first_prod_id}/frequently-bought-together?limit=3")
    assert fbt_res.status_code == 200
    print("  Frequently bought together: OK")

    print("--- 5. Testing Auth Flow ---")
    test_email = "tester_bandit@smartcart.ai"
    from app.database import users_collection
    users_collection.delete_one({"email": test_email})

    reg_res = client.post("/api/auth/register", json={
        "name": "Bandit Researcher",
        "email": test_email,
        "password": "SecurePassword123!",
        "preferred_categories": ["Electronics", "Fashion"]
    })
    assert reg_res.status_code == 201, f"Register failed: {reg_res.text}"
    token = reg_res.json()["access_token"]
    print("  Register: OK")

    login_res = client.post("/api/auth/login", json={
        "email": test_email,
        "password": "SecurePassword123!"
    })
    assert login_res.status_code == 200
    print("  Login: OK")

    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["user"]["email"] == test_email
    print("  Get current user: OK")

    print("--- 6. Testing Cart & Wishlist ---")
    cart_add = client.post("/api/cart/add", json={"product_id": first_prod_id, "quantity": 2}, headers=headers)
    assert cart_add.status_code == 200
    cart_get = client.get("/api/cart/", headers=headers)
    assert cart_get.status_code == 200
    assert len(cart_get.json()["items"]) == 1
    print("  Cart Add & Get: OK")

    wish_add = client.post("/api/wishlist/add", json={"product_id": first_prod_id}, headers=headers)
    assert wish_add.status_code == 200
    print("  Wishlist Add: OK")

    print("--- 7. Testing Order Placement ---")
    order_res = client.post("/api/orders/", json={
        "shipping_address": {
            "full_name": "Bandit Researcher",
            "street_address": "123 AI Boulevard",
            "city": "Bengaluru",
            "state": "Karnataka",
            "postal_code": "560001"
        }
    }, headers=headers)
    assert order_res.status_code == 201, f"Order failed: {order_res.text}"
    order_id = order_res.json()["order"]["order_id"]
    print(f"  Order placed: OK (Order ID: {order_id})")

    print("--- 8. Testing Behaviour Tracking ---")
    ev_res = client.post("/api/events/", json={
        "event_type": "product_click",
        "product_id": first_prod_id,
        "metadata": {"source": "test_suite"}
    }, headers=headers)
    assert ev_res.status_code == 201
    print("  Track product_click event: OK")

    print("--- 9. Testing LinUCB Recommendations & Feedback ---")
    rec_res = client.get("/api/recommendations/?limit=4", headers=headers)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert len(rec_data["recommendations"]) > 0
    rec_id = rec_data["recommendation_id"]
    first_rec = rec_data["recommendations"][0]
    print(f"  Generated {len(rec_data['recommendations'])} recommendations with LinUCB.")
    print(f"    Top item: '{first_rec['title']}' (UCB score: {first_rec['ucb_score']})")
    print(f"    Explanation: '{first_rec['explanation']}'")

    fb_res = client.post("/api/recommendations/feedback", json={
        "recommendation_id": rec_id,
        "product_id": first_rec["product_id"],
        "action": "like"
    }, headers=headers)
    assert fb_res.status_code == 200
    print(f"  Submitted feedback 'like': OK (Policy update: {fb_res.json()['policy_update']})")

    print("--- 10. Testing Admin Endpoints ---")
    admin_stats = client.get("/api/admin/stats")
    assert admin_stats.status_code == 200
    print(f"  Admin stats: OK (Active products: {admin_stats.json()['total_active_products']:,})")

    new_prod = client.post("/api/admin/products", json={
        "title": "Quantum Smart Watch Pro Test",
        "description": "High precision smartwatch with sapphire glass and AMOLED display.",
        "brand": "Noise",
        "category": "Electronics",
        "subcategory": "Smartwatches",
        "price": 4999.0,
        "stock_quantity": 40
    })
    assert new_prod.status_code == 200
    created_id = new_prod.json()["product"]["product_id"]
    print(f"  Admin create product: OK (New ID: {created_id})")

    inv_patch = client.patch(f"/api/admin/products/{created_id}/inventory", json={"stock_quantity": 85})
    assert inv_patch.status_code == 200
    assert inv_patch.json()["stock_quantity"] == 85
    print("  Admin inventory patch: OK")

    del_res = client.delete(f"/api/admin/products/{created_id}")
    assert del_res.status_code == 200
    print("  Admin delete product: OK")

    print("--- 11. Testing Analytics & Simulation ---")
    overview = client.get("/api/analytics/overview")
    assert overview.status_code == 200
    print(f"  Overview Analytics: OK (Total products: {overview.json()['total_products']:,})")

    print("\n=======================================================")
    print(" ALL 11 BACKEND API, LINUCB BANDIT & ADMIN TESTS PASSED!")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()

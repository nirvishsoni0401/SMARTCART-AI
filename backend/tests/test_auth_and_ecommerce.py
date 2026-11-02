import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.models import Product, User

client = TestClient(app)

def test_root_and_health():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json()["status"] == "online"

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_auth_flow():
    # 1. Register a new user
    import time
    user_email = f"test_{int(time.time())}@example.com"
    reg_res = client.post("/auth/register", json={
        "name": "Test Shopper",
        "email": user_email,
        "password": "securepassword123",
        "consent_tracking": True
    })
    assert reg_res.status_code == 201
    data = reg_res.json()
    assert "access_token" in data
    assert data["user"]["email"] == user_email
    token = data["access_token"]

    # 2. Duplicate registration should fail
    dup_res = client.post("/auth/register", json={
        "name": "Duplicate User",
        "email": user_email,
        "password": "securepassword123"
    })
    assert dup_res.status_code == 400

    # 3. Login
    login_res = client.post("/auth/login", json={
        "email": user_email,
        "password": "securepassword123"
    })
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

    # 4. Get Current User profile
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["name"] == "Test Shopper"

def test_products_catalog():
    # 1. List products
    res = client.get("/products?page=1&page_size=10")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 600
    assert len(data["products"]) == 10

    # 2. Filter by category
    cat_res = client.get("/products?category=Electronics&page_size=5")
    assert cat_res.status_code == 200
    cat_data = cat_res.json()
    assert all(p["category"] == "Electronics" for p in cat_data["products"])

    # 3. Filter by price
    price_res = client.get("/products?min_price=1000&max_price=3000&page_size=5")
    assert price_res.status_code == 200
    price_data = price_res.json()
    assert all(1000 <= p["price"] <= 3000 for p in price_data["products"])

    # 4. Sort price asc
    sort_res = client.get("/products?sort_by=price_asc&page_size=10")
    assert sort_res.status_code == 200
    prices = [p["price"] for p in sort_res.json()["products"]]
    assert prices == sorted(prices)

    # 5. Product detail
    first_id = data["products"][0]["id"]
    det_res = client.get(f"/products/{first_id}")
    assert det_res.status_code == 200
    assert det_res.json()["id"] == first_id

    # 6. Similar products
    sim_res = client.get(f"/products/{first_id}/similar")
    assert sim_res.status_code == 200
    assert isinstance(sim_res.json(), list)

def test_cart_and_checkout_flow():
    # Login as demo user
    login_res = client.post("/auth/login", json={
        "email": "rahul@smartcart.ai",
        "password": "user123"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Clear cart first
    client.delete("/cart", headers=headers)

    # Add item to cart
    add_res = client.post("/cart", json={"product_id": 1, "quantity": 2}, headers=headers)
    assert add_res.status_code == 201
    cart_data = add_res.json()
    assert cart_data["total_items"] >= 2

    # Checkout
    order_res = client.post("/orders", json={
        "shipping_address": {
            "name": "Rahul Sharma",
            "street": "123 MG Road",
            "city": "Bengaluru",
            "state": "Karnataka",
            "pin": "560001"
        },
        "payment_method": "UPI"
    }, headers=headers)
    assert order_res.status_code == 201
    order_data = order_res.json()
    assert order_data["status"] == "placed"
    assert len(order_data["items"]) >= 1
    order_id = order_data["id"]
    item_id = order_data["items"][0]["id"]

    # Verify cart is empty after checkout
    get_cart_res = client.get("/cart", headers=headers)
    assert get_cart_res.json()["total_items"] == 0

    # Test Item Return
    return_res = client.post(f"/orders/{order_id}/items/{item_id}/return", json={"reason": "Wrong size"}, headers=headers)
    assert return_res.status_code == 200
    updated_items = return_res.json()["items"]
    assert any(it["id"] == item_id and it["status"] == "returned" for it in updated_items)

def test_wishlist_flow():
    login_res = client.post("/auth/login", json={
        "email": "priya@smartcart.ai",
        "password": "user123"
    })
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Add to wishlist
    add_w = client.post("/wishlist", json={"product_id": 5}, headers=headers)
    assert add_w.status_code == 201
    assert any(item["product_id"] == 5 for item in add_w.json()["items"])

    # Remove from wishlist
    del_w = client.delete("/wishlist/5", headers=headers)
    assert del_w.status_code == 200
    assert not any(item["product_id"] == 5 for item in del_w.json()["items"])


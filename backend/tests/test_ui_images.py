"""
SmartCart AI — UI Image Health, Deterministic Covers, and API Contract Tests
Verifies that:
1. Every active catalog product has a non-empty string image_url.
2. The /static/covers/ route serves SVGs with HTTP 200 and image/svg+xml.
3. The /products/count endpoint returns the active product count.
4. Book cover generation is 100% idempotent and deterministic via SHA-256.
5. Cover dimensions maintain the required 3:4 aspect ratio with readable tags.
"""

import os
import sys
import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models import Product
from app.main import app
from app.services.cover_generator import (
    generate_book_cover_svg,
    save_book_cover,
    get_book_cover_filename,
    COVERS_DIR
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_every_active_product_has_non_empty_image_url(client: TestClient):
    """Verifies that every active catalog product has a valid non-empty string image_url."""
    resp = client.get("/products?page_size=100&page=1")
    assert resp.status_code == 200
    data = resp.json()
    total_active = data["total"]
    assert total_active == 262, f"Expected 262 active catalog products, got {total_active}"

    checked_count = 0
    total_pages = data["total_pages"]
    for page in range(1, total_pages + 1):
        p_resp = client.get(f"/products?page_size=100&page={page}")
        assert p_resp.status_code == 200
        p_data = p_resp.json()
        for p in p_data["products"]:
            checked_count += 1
            assert isinstance(p["image_url"], str), f"Product #{p['id']} image_url is not a string"
            assert len(p["image_url"].strip()) > 0, f"Product #{p['id']} has empty image_url"

            # Books must have local static cover
            if p["category"] == "Books":
                assert p["image_url"].startswith("/static/covers/"), (
                    f"Book #{p['id']} ({p['name']}) has non-local image_url: {p['image_url']}"
                )

    assert checked_count == 262


def test_covers_route_returns_200_with_svg_content_type(client: TestClient):
    """Verifies that the /static/covers/ endpoint returns HTTP 200 with image/svg+xml."""
    # Ensure at least one cover exists
    sample_cover = save_book_cover("Grokking Machine Learning & Bandit Algorithms", "Penguin India", "Computer Science", 526)
    assert sample_cover.startswith("/static/covers/")

    resp = client.get(sample_cover)
    assert resp.status_code == 200
    content_type = resp.headers.get("content-type", "")
    assert "image/svg+xml" in content_type or "svg" in content_type
    assert len(resp.content) > 500
    assert resp.text.startswith("<svg")
    assert "</svg>" in resp.text


def test_product_api_returns_image_url_as_string_and_count(client: TestClient):
    """Verifies that GET /products/count and GET /products/{id} conform to API contracts."""
    # Test count endpoint
    resp = client.get("/products/count")
    assert resp.status_code == 200
    count_data = resp.json()
    assert "total" in count_data
    assert "active" in count_data
    assert count_data["total"] == 262
    assert count_data["active"] == 262

    # Test single product detail
    prod_resp = client.get("/products/526")
    assert prod_resp.status_code == 200
    prod_data = prod_resp.json()
    assert isinstance(prod_data["image_url"], str)
    assert prod_data["image_url"].startswith("/static/covers/")
    assert prod_data["category"] == "Books"


def test_seed_idempotency_with_covers():
    """Verifies that generating book covers twice yields byte-for-byte identical output."""
    title = "Designing Data-Intensive Applications"
    brand = "O'Reilly"
    subcat = "Technology"

    cover_1 = generate_book_cover_svg(title, brand, subcat)
    cover_2 = generate_book_cover_svg(title, brand, subcat)
    assert cover_1 == cover_2, "Book cover generation is non-deterministic!"

    path_1 = save_book_cover(title, brand, subcat, 528)
    path_2 = save_book_cover(title, brand, subcat, 528)
    assert path_1 == path_2

    disk_file = Path(__file__).resolve().parent.parent / path_1.lstrip("/")
    assert disk_file.is_file()
    assert disk_file.read_text(encoding="utf-8") == cover_1


def test_cover_aspect_ratio_and_contrast():
    """Verifies that generated book covers adhere to the 3:4 aspect ratio and contain required badges."""
    svg = generate_book_cover_svg("Atomic Habits: Easy & Proven Way to Build Good Habits", "Jaico", "Self-Help")
    assert 'viewBox="0 0 600 800"' in svg, "Cover does not have 3:4 aspect ratio (600x800)"
    assert ">BOOK</text>" in svg, "Missing 'BOOK' tag badge"
    assert "Jaico" in svg, "Missing publisher/brand label"
    assert "Atomic Habits" in svg, "Missing book title in SVG"
    assert "SmartCart Library Edition" in svg, "Missing imprint line"

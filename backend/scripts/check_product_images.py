"""
SmartCart AI — Product Image Health Check Script
Diagnoses reachability, response times, HTTP statuses, and error types
for all active catalog product images (primary and gallery).
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
OUTPUT_PATH = ROOT_DIR / "experiments" / "results" / "image_check.json"

USER_AGENT = "SmartCart-ImageChecker/1.0 (+http://smartcart.ai)"

def check_url(url: str, timeout: float = 4.0, max_retries: int = 1) -> dict:
    """
    Checks reachability of a single URL or local path.
    Tries HEAD first; falls back to GET with stream=True on 405/failure.
    Handles relative / local paths (/static/...) by checking disk file.
    """
    if not url or not isinstance(url, str):
        return {"status": "invalid_url", "error_type": "empty_or_non_string", "code": None, "host": None}

    url = url.strip()
    # Check if local relative static path or local absolute
    if url.startswith("/static/") or url.startswith("static/"):
        rel_path = url.lstrip("/")
        local_file = ROOT_DIR / "backend" / rel_path
        if local_file.is_file() and local_file.stat().st_size > 0:
            return {
                "status": "ok",
                "error_type": None,
                "code": 200,
                "host": "localhost_static",
                "content_type": "image/svg+xml" if local_file.suffix == ".svg" else "image/png",
                "is_local": True
            }
        else:
            return {
                "status": "failed",
                "error_type": "local_file_not_found",
                "code": 404,
                "host": "localhost_static",
                "is_local": True
            }

    parsed = urlparse(url)
    host = parsed.netloc or "unknown"
    if not parsed.scheme or not parsed.netloc:
        return {"status": "invalid_url", "error_type": "invalid_url_format", "code": None, "host": host}

    # If pointing to localhost
    if parsed.hostname in ("127.0.0.1", "localhost"):
        if parsed.path.startswith("/static/"):
            rel_path = parsed.path.lstrip("/")
            local_file = ROOT_DIR / "backend" / rel_path
            if local_file.is_file() and local_file.stat().st_size > 0:
                return {
                    "status": "ok",
                    "error_type": None,
                    "code": 200,
                    "host": host,
                    "content_type": "image/svg+xml" if local_file.suffix == ".svg" else "image/png",
                    "is_local": True
                }

    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries + 1):
        try:
            # Try HEAD first
            try:
                resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
                if resp.status_code == 200:
                    ct = resp.headers.get("content-type", "")
                    return {
                        "status": "ok",
                        "error_type": None,
                        "code": resp.status_code,
                        "host": host,
                        "content_type": ct,
                        "is_local": False
                    }
                elif resp.status_code in (405, 403, 400):
                    # Some CDNs disallow HEAD; fall back to GET with stream=True
                    pass
                else:
                    return {
                        "status": "failed",
                        "error_type": f"http_{resp.status_code}",
                        "code": resp.status_code,
                        "host": host,
                        "is_local": False
                    }
            except (requests.exceptions.RequestException, Exception):
                pass

            # Fallback to GET
            resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "")
                return {
                    "status": "ok",
                    "error_type": None,
                    "code": resp.status_code,
                    "host": host,
                    "content_type": ct,
                    "is_local": False
                }
            else:
                return {
                    "status": "failed",
                    "error_type": f"http_{resp.status_code}",
                    "code": resp.status_code,
                    "host": host,
                    "is_local": False
                }

        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return {"status": "failed", "error_type": "timeout", "code": None, "host": host, "is_local": False}
        except requests.exceptions.ConnectionError as e:
            err_str = str(e)
            if "getaddrinfo" in err_str or "NameResolution" in err_str:
                return {"status": "failed", "error_type": "dns_failure", "code": None, "host": host, "is_local": False}
            if attempt == max_retries:
                return {"status": "failed", "error_type": "connection_error", "code": None, "host": host, "is_local": False}
        except Exception as e:
            if attempt == max_retries:
                return {"status": "failed", "error_type": f"error_{type(e).__name__}", "code": None, "host": host, "is_local": False}

        time.sleep(0.2)

    return {"status": "failed", "error_type": "unknown_failure", "code": None, "host": host, "is_local": False}


def main():
    parser = argparse.ArgumentParser(description="Check product image availability")
    parser.add_argument("--workers", type=int, default=25, help="Worker threads")
    parser.add_argument("--timeout", type=float, default=3.5, help="Request timeout in seconds")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, name, category, brand, source, image_url, attributes
        FROM products
        WHERE is_active = 1
        ORDER BY id ASC
    """)
    rows = c.fetchall()
    conn.close()

    print(f"Fetched {len(rows)} active products from database.")

    products_to_check = []
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
        products_to_check.append({
            "id": pid,
            "name": name,
            "department": cat,
            "brand": brand,
            "source": src,
            "image_url": img_url,
            "gallery": gallery
        })

    print(f"Beginning concurrent check of {len(products_to_check)} products ({args.workers} workers, {args.timeout}s timeout)...")
    start_time = time.time()

    checked_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(check_url, p["image_url"], args.timeout): p
            for p in products_to_check
        }
        for future in as_completed(future_map):
            p = future_map[future]
            res = future.result()
            checked_results.append({
                "product_id": p["id"],
                "name": p["name"],
                "department": p["department"],
                "source": p["source"],
                "image_url": p["image_url"],
                "check_result": res
            })

    elapsed = time.time() - start_time
    print(f"Completed image checks in {elapsed:.2f}s.\n")

    # Aggregate summaries by source and by department
    by_source = {}
    by_dept = {}
    hosts_count = {}
    error_types_count = {}

    total_ok = 0
    total_failed = 0

    for item in checked_results:
        src = item["source"]
        dept = item["department"]
        res = item["check_result"]
        ok = (res["status"] == "ok")
        host = res.get("host") or "unknown"
        err = res.get("error_type") or "none"

        if ok:
            total_ok += 1
        else:
            total_failed += 1
            error_types_count[err] = error_types_count.get(err, 0) + 1

        hosts_count[host] = hosts_count.get(host, 0) + 1

        # by source
        if src not in by_source:
            by_source[src] = {"total": 0, "ok": 0, "failed": 0, "error_types": {}}
        by_source[src]["total"] += 1
        if ok:
            by_source[src]["ok"] += 1
        else:
            by_source[src]["failed"] += 1
            by_source[src]["error_types"][err] = by_source[src]["error_types"].get(err, 0) + 1

        # by department
        if dept not in by_dept:
            by_dept[dept] = {"total": 0, "ok": 0, "failed": 0, "source": src, "error_types": {}}
        by_dept[dept]["total"] += 1
        if ok:
            by_dept[dept]["ok"] += 1
        else:
            by_dept[dept]["failed"] += 1
            by_dept[dept]["error_types"][err] = by_dept[dept]["error_types"].get(err, 0) + 1

    # Print summary tables
    print("=" * 80)
    print("SMARTCART AI — PRODUCT IMAGE HEALTH DIAGNOSIS TABLE")
    print("=" * 80)
    print("\n1. SUMMARY BY SOURCE:")
    print(f"{'Source':<15} | {'Total':<8} | {'OK':<8} | {'Failed':<8} | {'Pass Rate':<10} | {'Primary Errors'}")
    print("-" * 80)
    for src, stats in sorted(by_source.items()):
        pass_rate = (stats["ok"] / stats["total"] * 100) if stats["total"] > 0 else 0
        err_str = ", ".join(f"{k}:{v}" for k, v in stats["error_types"].items()) if stats["error_types"] else "None"
        print(f"{src:<15} | {stats['total']:<8} | {stats['ok']:<8} | {stats['failed']:<8} | {pass_rate:>8.1f}% | {err_str}")

    print("\n2. SUMMARY BY DEPARTMENT (CATEGORY):")
    print(f"{'Department':<18} | {'Source':<10} | {'Total':<6} | {'OK':<6} | {'Failed':<6} | {'Pass Rate':<10} | {'Error Types'}")
    print("-" * 80)
    for dept, stats in sorted(by_dept.items()):
        pass_rate = (stats["ok"] / stats["total"] * 100) if stats["total"] > 0 else 0
        err_str = ", ".join(f"{k}:{v}" for k, v in stats["error_types"].items()) if stats["error_types"] else "None"
        print(f"{dept:<18} | {stats['source']:<10} | {stats['total']:<6} | {stats['ok']:<6} | {stats['failed']:<6} | {pass_rate:>8.1f}% | {err_str}")

    print("\n3. HOSTS INVOLVED:")
    for h, cnt in sorted(hosts_count.items(), key=lambda x: -x[1]):
        print(f"  - {h}: {cnt} images")

    print("\n4. OVERALL TOTALS:")
    overall_pass_rate = (total_ok / len(checked_results) * 100) if checked_results else 0
    print(f"  Total Active Products: {len(checked_results)}")
    print(f"  Total OK:              {total_ok} ({overall_pass_rate:.1f}%)")
    print(f"  Total Failed:          {total_failed} ({100 - overall_pass_rate:.1f}%)")
    print("=" * 80)

    # Save to experiments/results/image_check.json
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_active_products": len(checked_results),
        "total_ok": total_ok,
        "total_failed": total_failed,
        "overall_pass_rate_pct": round(overall_pass_rate, 2),
        "by_source": by_source,
        "by_department": by_dept,
        "hosts": hosts_count,
        "error_types": error_types_count,
        "details": checked_results
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved diagnosis results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

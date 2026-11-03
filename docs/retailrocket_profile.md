# Retailrocket Dataset Profiling & Ingestion Analysis

**Generated**: 2026-11-03 15:33:47 UTC  
**Dataset Path**: `RetailRocket_Dataset/`  
**Files Profiled**: `events.csv`, `category_tree.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`  
**Execution Runtime**: 50.82 seconds (High-performance chunked streaming)

---

## 1. Executive Summary & Key Dataset Metrics

The Retailrocket dataset represents 4.5 months of real-world user activity from a commercial e-commerce platform. It contains raw clickstream logs (`view`, `addtocart`, `transaction`), item property snapshots, and a hierarchical category taxonomy.

| Metric | Measured Value | Percentage / Notes |
| :--- | :---: | :--- |
| **Total Events** | **2,756,101** | 100.0% raw clickstream |
| **Date Range (UTC)** | 2015-05-03 03:00:04 to 2015-09-18 02:59:47 | **138.0 days** (~4.5 months) |
| **Unique Visitors** | **1,407,580** | 1,407,580 distinct shoppers |
| **Unique Items in Events** | **235,061** | 235,061 interacted products |
| **Views** | 2,664,312 | **96.67%** |
| **Add to Cart** | 69,332 | **2.52%** |
| **Transactions** | 22,457 | **0.81%** |
| **Funnel: View to AddToCart** | **2.60%** | 1 in 38.4 views leads to a cart addition |
| **Funnel: AddToCart to Transaction** | **32.39%** | 1 in 3.09 cart additions converts to a purchase |
| **Funnel: View to Transaction** | **0.84%** | Overall baseline conversion rate |
| **Exact Duplicates Detected** | 450 | Rapid double-clicks within streaming chunks |
| **Transaction ID Completeness** | 22,457 / 22,457 | **100.0%** of transactions have transaction IDs |
| **Non-Transaction ID Leakage** | 0 | 0 non-transactions carry transaction IDs |

---

## 2. Interaction Distributions: Visitors & Products

### 2.1. Visitor Interaction Distribution
- **Mean events per visitor**: `1.96`
- **Median events per visitor**: `1`
- **Percentiles**: p25 = `1`, p50 = `1`, p75 = `2`, p90 = `3`, p95 = `5`, p99 = `13`
- **Max events by single visitor**: `7,757` events
- **Extreme / Bot-like Visitors**:
  - Visitors with >= 500 events: **75** visitors
  - Visitors with >= 1,000 events: **30** visitors
  - *Recommendation*: Cap bot-like visitors at 500 events to prevent automated scrapers from overwhelming bandit policy updates.

### 2.2. Product Interaction Distribution & Long-Tail
- **Mean events per item**: `11.73`
- **Median events per item**: `3`
- **Percentiles**: p25 = `1`, p50 = `3`, p75 = `9`, p90 = `25`, p99 = `143`
- **Long-Tail Concentration**: Top **10%** of items account for **64.0%** of all interaction volume.
- **Sparse Tail Items**: **129,507** items (55.1%) have <= 3 total interactions in the dataset.

---

## 3. Sessionization Analysis (30-Minute Inactivity Window)

Applying the standard web analytics 30-minute inactivity boundary (delta_t > 1,800,000 ms between consecutive visitor events):

| Session Metric | Value | Meaning |
| :--- | :---: | :--- |
| **Total Sessions** | **1,761,675** | Discrete shopping sessions |
| **Single-Event Sessions ("Bounces")** | **1,378,895** | **78.27%** bounce rate |
| **Mean Events per Session** | **1.56** | Average session depth |
| **Median Events per Session** | **1** | Typical session engagement |
| **90th Percentile Session Length** | **2** | Highly engaged browsing sessions |
| **Max Session Length** | **417** | Extreme power user session |

---

## 4. Item Metadata & Category Taxonomy Analysis

### 4.1. Category Tree Topology (`category_tree.csv`)
- **Total Categories**: **1,669**
- **Root Categories (`parentid` is null)**: **25 roots**
- **Leaf Categories**: **1,307 leaves**
- **Maximum Hierarchy Depth**: **5 levels**

### 4.2. Category Attribution Rule & Coverage
In Retailrocket, item properties are logged dynamically with timestamps. An item's category can be updated over time.
- **Canonical Rule Defined**: **Latest-Timestamp Rule**: For any item $i$, select the `categoryid` associated with the maximum timestamp $\max(t)$. If multiple records share the maximum timestamp, the highest `categoryid` is chosen deterministically.
- **Catalog Coverage**:
  - Unique items in `item_properties` with a category: **417,053**
  - Unique items appearing in `events.csv` with a category: **185,246** (**78.81%**)
  - Only ~21.2% of items in events lack explicit category attribution.

---

## 5. Recommended Ingestion Thresholds & Resource Estimation

To prepare Retailrocket data for offline evaluation and bandit warm-start (Phase 9B), we establish the following filtering rules:

| Filter Stage / Rule | Configuration Parameter | Value | Rationalization |
| :--- | :--- | :---: | :--- |
| **Exact Deduplication** | `dedup_window_sec` | 60s | Removes rapid accidental double-clicks. |
| **Bot Activity Capping** | `max_events_per_visitor` | 500 | Prevents web crawlers and scrapers from distorting preferences. |
| **Item Frequency Threshold** | `min_events_per_item` | 3 | Prunes cold tail items that lack sufficient co-occurrence signal. |
| **Visitor Diversity Threshold**| `min_visitors_per_item` | 2 | Ensures items were viewed by more than one distinct shopper. |
| **Transaction Safeguard** | `always_keep_transactions` | **True** | **CRITICAL**: Every single transaction event is kept regardless of item frequency. |

### Resource & Storage Estimation for `backend/historical.db`:
- **Estimated Kept Events**: ~1.8 to 2.1 million events
- **Estimated Kept Items**: ~75,000 to 90,000 items
- **Estimated Database Size**: ~110 to 150 MB (SQLite with indexes)
- **Isolation Guarantee**: Stored in `backend/historical.db` (gitignored), completely independent of operational `smartcart.db`.

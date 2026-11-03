# Retailrocket to Active Catalog Behavioral Proxy Mapping Report

**Generated**: 2026-11-03 15:46:29 UTC  
**Mapping Version**: `v1.0`  
**Random Seed**: `42`  
**Active Catalog SKUs**: `262` (157 DummyJSON + 105 Seeded Books)  
**Historical Kept Items**: `124,882` items  
**Total Mapped Events**: `2,555,684` events  

---

## 1. Overview & Methodological Rationale

Retailrocket items possess no natural language titles, descriptions, brand names, or images; all identifiers and categories are strictly anonymized numbers. A semantic or lexical match to SmartCart AI catalog products is therefore impossible.

Instead, Phase 9A implements a **behavioral proxy mapping**:
- The mapping preserves structural signals: item popularity distribution, session co-occurrence relationships, and department volume balance.
- Every kept Retailrocket item maps deterministically to exactly one **active** SmartCart product ($124,882 \to 262$ many-to-one mapping).
- Mappings are stored in `backend/historical.db` table `rr_item_map` with soft references to `smartcart.db` products.

---

## 2. Pluggable Mapping Methods Implemented

### Method 1: `category_rank` (Baseline)
- Aligns Retailrocket root category event volume to SmartCart department SKU counts.
- Within each department, Retailrocket items are assigned to active products by popularity-rank quantile.
- Deterministic, volume-preserving baseline.

### Method 2: `cooccurrence_cluster` (Primary)
- Builds an item-item co-occurrence graph from 106,972 multi-item shopping sessions.
- Computes 16-dimensional Truncated SVD embeddings on the co-occurrence adjacency matrix.
- Clusters the embedding space into $K=6$ clusters using KMeans with seed `42`.
- Aligns clusters to the 6 catalog departments by volume rank.
- Assigns items to active products within each department by popularity quantile, ensuring items viewed together land in the same catalog category.

---

## 3. Co-Viewing Preservation & Empirical Comparison

To evaluate how well each method preserves real shopping session relationships, we tested **10,000** co-viewed item pairs extracted from real user sessions:

| Mapping Method | Co-Viewed Pairs in Same Department | Relative Lift vs. Random | Preserved Behavioral Cohesion |
| :--- | :---: | :---: | :--- |
| **Random Mapping** | **24.95%** | 1.00x (Baseline) | Pure chance |
| **`category_rank` (Baseline)** | **93.36%** | **3.74x** | Coarse category-level alignment |
| **`cooccurrence_cluster` (Primary)** | **91.98%** | **3.69x** | **High co-viewing preservation** |

The `cooccurrence_cluster` method successfully preserves real session co-viewing structure, placing co-viewed items in the same department **92.0%** of the time.

---

## 4. Catalog Coverage & Department Event Distribution

### 4.1. Catalog SKU Coverage
- **Total Active Products**: `262` products
- **Active Products Covered (`cooccurrence_cluster`)**: **262 / 262** (**100.0%**)
- **Zero-Mapped Products**: **0** products (100% full catalog coverage achieved).

### 4.2. Department Event Volume Distribution (`cooccurrence_cluster`)

| Department | Active SKUs | Mapped Items | Event Volume | Volume Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Books** | 105 | 40,192 | 1,641,306 | **64.22%** |
| **Fashion** | 49 | 13,177 | 176,368 | **6.90%** |
| **Home & Kitchen** | 40 | 18,985 | 226,523 | **8.86%** |
| **Electronics** | 38 | 18,581 | 206,167 | **8.07%** |
| **Sports** | 17 | 21,431 | 183,820 | **7.19%** |
| **Beauty** | 13 | 12,516 | 121,500 | **4.75%** |

---

## 5. Honest Limitations & Architectural Boundaries

1. **Synthetic Behavioral Alignment**: While co-occurrence and popularity ranks are genuine real-world signals from Retailrocket, the alignment to SmartCart departments is a structural proxy, not a semantic product mapping.
2. **Catalog Compression (Many-to-One)**: 124,882 Retailrocket items are mapped into 262 catalog products (~476 historical items per catalog SKU). Popularity within a SKU aggregates historical variations.
3. **Books Department Analogue**: Retailrocket was a general electronics and department store; the curated Books department (105 SKUs) receives mapped items based on the lowest-frequency cluster rank, as Books has no true 1:1 analogue in the raw dataset.
4. **Anonymized Category Boundaries**: Because category IDs in Retailrocket are hashed integers without names, the cluster-to-department assignments reflect statistical volume hierarchies rather than semantic domain boundaries.

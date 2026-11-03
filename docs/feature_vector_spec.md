# SmartCart AI — Feature Vector Specification (R^36)

**Specification Version**: `v1.0`  
**Feature Dimension**: $d = 36$  
**Canonical Specification Hash**: `75f60ed70b2a`  
**Binding Scope**: Live Recommender Pipeline, Experiment Simulator, Offline Evaluation, and Phase 9B Warm-Start.

---

## 1. Overview & Mathematical Definition

Every recommendation decision evaluates a joint context-item feature vector:
$$\phi(u, s, p) \in \mathbb{R}^{36}$$
where $u$ is the user interaction context, $s$ is the session context, and $p$ is the product candidate.

The LinUCB bandit scoring function calculates the upper confidence bound as:
$$\text{UCB}(p) = x^T \hat{\theta} + \alpha \sqrt{x^T A^{-1} x}$$
where $x = \phi(u, s, p)$, $A = I_{36} + \sum_{\tau} x_\tau x_\tau^T$, $b = \sum_{\tau} r_\tau x_\tau$, and $\hat{\theta} = A^{-1} b$.

All models, persistence layers, and future Phase 9B warm-start weights must strictly conform to this 36-dimensional layout and match `FEATURE_SPEC_HASH = 75f60ed70b2a`.

---

## 2. Dimension Index Layout

| Index | Feature Name | Range / Type | Mathematical Definition & Scaling | Cold-Start Default |
| :---: | :--- | :---: | :--- | :---: |
| **0** | `affinity_electronics` | $[0.0, 1.0]$ | Time-decayed interaction weight for Electronics: $\sum_i w(t_i) \cdot \mathbb{I}(\text{cat}=\text{Electronics}) / \sum_i w(t_i)$ | $0.0$ |
| **1** | `affinity_fashion` | $[0.0, 1.0]$ | Time-decayed interaction weight for Fashion: $\sum_i w(t_i) \cdot \mathbb{I}(\text{cat}=\text{Fashion}) / \sum_i w(t_i)$ | $0.0$ |
| **2** | `affinity_home_kitchen` | $[0.0, 1.0]$ | Time-decayed interaction weight for Home & Kitchen | $0.0$ |
| **3** | `affinity_beauty` | $[0.0, 1.0]$ | Time-decayed interaction weight for Beauty & Personal Care | $0.0$ |
| **4** | `affinity_sports` | $[0.0, 1.0]$ | Time-decayed interaction weight for Sports & Fitness | $0.0$ |
| **5** | `affinity_books` | $[0.0, 1.0]$ | Time-decayed interaction weight for Books | $0.0$ |
| **6** | `affinity_brand` | $[0.0, 1.0]$ | Interaction share with candidate's brand: $N_{\text{brand}} / N_{\text{total}}$ | $0.0$ |
| **7** | `user_avg_price_level` | $[0.0, 1.0]$ | Log-normalized mean historical price: $\ln(1 + \bar{P}_{\text{user}}) / \ln(1 + 200000)$ | $0.35$ ($\approx \text{₹}2,000$) |
| **8** | `session_budget_cap` | $[0.0, 1.0]$ | Log-normalized session budget: $\ln(1 + B_{\text{session}}) / \ln(1 + 200000)$ | Equals feature 7 |
| **9** | `price_fit_score` | $[0.0, 1.0]$ | Gaussian/linear proximity: $\max(0, 1 - |\ln(1+P_p) - \ln(1+B)| / 2.0)$ | $0.5$ |
| **10** | `in_budget_flag` | $\{0.0, 1.0\}$ | Binary indicator: $\mathbb{I}(P_p \le B_{\text{session}})$ | $1.0$ |
| **11** | `category_search_match` | $\{0.0, 1.0\}$ | Binary match between candidate category and active session search query or category filter | $0.0$ |
| **12** | `cat_onehot_electronics` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Electronics | Candidate specific |
| **13** | `cat_onehot_fashion` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Fashion | Candidate specific |
| **14** | `cat_onehot_home_kitchen`| $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Home & Kitchen | Candidate specific |
| **15** | `cat_onehot_beauty` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Beauty | Candidate specific |
| **16** | `cat_onehot_sports` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Sports | Candidate specific |
| **17** | `cat_onehot_books` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Books | Candidate specific |
| **18** | `product_price_norm` | $[0.0, 1.0]$ | Log-scaled product price: $\ln(1 + P_p) / \ln(1 + 200000)$ | Candidate specific |
| **19** | `product_rating_norm` | $[0.0, 1.0]$ | Normalized star rating: $(R_p - 1.0) / 4.0$ | Candidate specific |
| **20** | `product_discount_pct` | $[0.0, 1.0]$ | Linear discount percentage: $\text{discount\_pct} / 100.0$ | Candidate specific |
| **21** | `product_popularity` | $[0.0, 1.0]$ | Normalized popularity score: $\text{popularity\_score} / 100.0$ | Candidate specific |
| **22** | `brand_hash_bin_0` | $\{0.0, 1.0\}$ | Hash bucket 0 of product brand name: $\mathbb{I}(|\text{hash}(\text{brand})| \pmod 8 == 0)$ | Candidate specific |
| **23** | `brand_hash_bin_1` | $\{0.0, 1.0\}$ | Hash bucket 1 of product brand name | Candidate specific |
| **24** | `brand_hash_bin_2` | $\{0.0, 1.0\}$ | Hash bucket 2 of product brand name | Candidate specific |
| **25** | `brand_hash_bin_3` | $\{0.0, 1.0\}$ | Hash bucket 3 of product brand name | Candidate specific |
| **26** | `brand_hash_bin_4` | $\{0.0, 1.0\}$ | Hash bucket 4 of product brand name | Candidate specific |
| **27** | `brand_hash_bin_5` | $\{0.0, 1.0\}$ | Hash bucket 5 of product brand name | Candidate specific |
| **28** | `brand_hash_bin_6` | $\{0.0, 1.0\}$ | Hash bucket 6 of product brand name | Candidate specific |
| **29** | `brand_hash_bin_7` | $\{0.0, 1.0\}$ | Hash bucket 7 of product brand name | Candidate specific |
| **30** | `new_product_flag` | $\{0.0, 1.0\}$ | Binary indicator: $\mathbb{I}(\text{product\_id} > 550)$ | Candidate specific |
| **31** | `user_activity_level` | $[0.0, 1.0]$ | Sub-linear interaction volume: $\min(1.0, \ln(1 + N_{\text{events}}) / \ln(50))$ | $0.0$ |
| **32** | `cart_intent_match` | $[0.0, 1.0]$ | Share of items currently in cart belonging to this candidate's category | $0.0$ |
| **33** | `high_rating_boost` | $\{0.0, 1.0\}$ | High satisfaction heuristic indicator: $\mathbb{I}(R_p \ge 4.5)$ | Candidate specific |
| **34** | `high_discount_boost` | $\{0.0, 1.0\}$ | Bargain seeker heuristic indicator: $\mathbb{I}(\text{discount\_pct} \ge 30\%)$ | Candidate specific |
| **35** | `intercept_bias` | $1.0$ | Constant bias term representing baseline preference expectation | $1.0$ |

---

## 3. Candidate Generation & Pipeline Parameters

The live recommendation pipeline operates in three successive stages:

1. **Stage 1: Candidate Generation** (`app/services/candidate_generation.py`)
   - `CANDIDATE_POOL_SIZE`: **50 candidates** maximum sampled from active catalog ($262$ SKUs).
   - `MIN_CANDIDATES_PER_CATEGORY`: **2 items** minimum per department to guarantee cross-category diversity.
   - Exclusion filters: Inactive items (`is_active == False`), out-of-stock items (`stock <= 0`), and items purchased in user's last 3 completed orders.
2. **Stage 2: Algorithmic Scoring** (`app/recommender/`)
   - Computes UCB score or policy ranking using feature vector $\phi \in \mathbb{R}^{36}$.
   - LinUCB confidence parameter: $\alpha = 0.5$ (configurable).
3. **Stage 3: Diversity & MMR Re-Ranking** (`app/services/diversity.py`)
   - `DEFAULT_RECOMMENDATION_COUNT` ($K$): **10 products**.
   - `MAX_PER_CATEGORY_TOP10`: **3 items** maximum per department in top-10.
   - `MMR_LAMBDA`: **0.65** balance between score exploitation ($0.65$) and category/brand redundancy penalty ($0.35$).
   - `IMPRESSION_TIMEOUT_MINUTES`: **5 minutes** for resolving unclicked impressions as skips.

---

## 4. Verification & Model Compliance Hash

The canonical specification string:
```
dim=36:0-5=user_category_affinity(6):6=user_brand_affinity:7=user_avg_price_log:8=session_budget_log:9=price_fit_score:10=in_budget_flag:11=category_search_match:12-17=product_category_onehot(6):18=product_price_log:19=product_rating_norm:20=product_discount_pct:21=product_popularity_score:22-29=brand_hash_bucket(8):30=new_product_flag:31=user_activity_level:32=cart_intent_match:33=high_rating_flag:34=high_discount_flag:35=intercept_bias
```
Produces SHA-256 digest prefix:
```
FEATURE_SPEC_HASH = "75f60ed70b2a"
```
Before loading any model weights in Phase 9B, the model store must verify that `model.d == 36` and assert compatibility against `FEATURE_SPEC_HASH`.

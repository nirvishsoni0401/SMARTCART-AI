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

All models, persistence layers, and Phase 9B warm-start weights strictly conform to this 36-dimensional layout and match `FEATURE_SPEC_HASH = 75f60ed70b2a`. Live and historical feature extraction share the exact same underlying implementation in `app.services.feature_engineering:build_context_from_history`.

---

## 2. Dimension Index Layout & History Availability

| Index | Feature Name | Range / Type | Mathematical Definition & Scaling | Cold-Start Default | History Availability (Retailrocket) |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **0** | `affinity_electronics` | $[0.0, 1.0]$ | Time-decayed interaction weight for Electronics | $0.0$ | **Shared Identically** (from past events $\exp(-0.23 \Delta t)$) |
| **1** | `affinity_fashion` | $[0.0, 1.0]$ | Time-decayed interaction weight for Fashion | $0.0$ | **Shared Identically** (from past events) |
| **2** | `affinity_home_kitchen` | $[0.0, 1.0]$ | Time-decayed interaction weight for Home & Kitchen | $0.0$ | **Shared Identically** (from past events) |
| **3** | `affinity_beauty` | $[0.0, 1.0]$ | Time-decayed interaction weight for Beauty | $0.0$ | **Shared Identically** (from past events) |
| **4** | `affinity_sports` | $[0.0, 1.0]$ | Time-decayed interaction weight for Sports | $0.0$ | **Shared Identically** (from past events) |
| **5** | `affinity_books` | $[0.0, 1.0]$ | Time-decayed interaction weight for Books | $0.0$ | **Shared Identically** (from past events) |
| **6** | `affinity_brand` | $[0.0, 1.0]$ | Interaction share with candidate's brand | $0.0$ | **Shared Identically** (from past mapped brand events) |
| **7** | `user_avg_price_level` | $[0.0, 1.0]$ | Log-normalized mean historical price: $\ln(1 + \bar{P}) / \ln(1 + 200000)$ | $0.35$ ($\approx \text{₹}1,800$) | **Shared Identically** (mean of prior cart_add/trans) |
| **8** | `session_budget_cap` | $[0.0, 1.0]$ | Log-normalized session budget: $\ln(1 + B) / \ln(1 + 200000)$ | Equals feature 7 | **Proxied** (set to user avg price / default ₹1,800) |
| **9** | `price_fit_score` | $[0.0, 1.0]$ | Proximity: $\max(0, 1 - \|\ln(1+P_p) - \ln(1+B)\| / 2.0)$ | $0.5$ | **Shared Identically** (computed against budget) |
| **10** | `in_budget_flag` | $\{0.0, 1.0\}$ | Binary indicator: $\mathbb{I}(P_p \le B_{\text{session}})$ | $1.0$ | **Shared Identically** (indicator against budget) |
| **11** | `category_search_match` | $\{0.0, 1.0\}$ | Match between category and session query/filter | $0.0$ | **Defaulted (0.0)** (no search queries in RR log) |
| **12** | `cat_onehot_electronics` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Electronics | Candidate specific | **Shared Identically** (catalog product category) |
| **13** | `cat_onehot_fashion` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Fashion | Candidate specific | **Shared Identically** (catalog product category) |
| **14** | `cat_onehot_home_kitchen`| $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Home & Kitchen | Candidate specific | **Shared Identically** (catalog product category) |
| **15** | `cat_onehot_beauty` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Beauty | Candidate specific | **Shared Identically** (catalog product category) |
| **16** | `cat_onehot_sports` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Sports | Candidate specific | **Shared Identically** (catalog product category) |
| **17** | `cat_onehot_books` | $\{0.0, 1.0\}$ | 1-hot candidate category indicator for Books | Candidate specific | **Shared Identically** (catalog product category) |
| **18** | `product_price_norm` | $[0.0, 1.0]$ | Log-scaled product price: $\ln(1 + P_p) / \ln(1 + 200000)$ | Candidate specific | **Shared Identically** (catalog product price) |
| **19** | `product_rating_norm` | $[0.0, 1.0]$ | Normalized star rating: $(R_p - 1.0) / 4.0$ | Candidate specific | **Proxied** (catalog product rating; RR has no stars) |
| **20** | `product_discount_pct` | $[0.0, 1.0]$ | Linear discount percentage: $\text{discount\_pct} / 100.0$ | Candidate specific | **Shared Identically** (catalog product discount) |
| **21** | `product_popularity` | $[0.0, 1.0]$ | Normalized popularity score: $\text{popularity\_score} / 100.0$ | Candidate specific | **Shared Identically** (catalog product popularity) |
| **22** | `brand_hash_bin_0` | $\{0.0, 1.0\}$ | Hash bucket 0 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **23** | `brand_hash_bin_1` | $\{0.0, 1.0\}$ | Hash bucket 1 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **24** | `brand_hash_bin_2` | $\{0.0, 1.0\}$ | Hash bucket 2 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **25** | `brand_hash_bin_3` | $\{0.0, 1.0\}$ | Hash bucket 3 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **26** | `brand_hash_bin_4` | $\{0.0, 1.0\}$ | Hash bucket 4 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **27** | `brand_hash_bin_5` | $\{0.0, 1.0\}$ | Hash bucket 5 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **28** | `brand_hash_bin_6` | $\{0.0, 1.0\}$ | Hash bucket 6 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **29** | `brand_hash_bin_7` | $\{0.0, 1.0\}$ | Hash bucket 7 of product brand name | Candidate specific | **Shared Identically** (catalog product brand) |
| **30** | `new_product_flag` | $\{0.0, 1.0\}$ | Binary indicator: $\mathbb{I}(\text{product\_id} > 550)$ | Candidate specific | **Defaulted (0.0)** (mapped to active catalog SKUs) |
| **31** | `user_activity_level` | $[0.0, 1.0]$ | Sub-linear interaction volume: $\min(1.0, \ln(1 + N) / \ln(50))$ | $0.0$ | **Shared Identically** (count of prior visitor events) |
| **32** | `cart_intent_match` | $[0.0, 1.0]$ | Share of items currently in cart in candidate's cat | $0.0$ | **Proxied** (from prior cart_add events in session) |
| **33** | `high_rating_boost` | $\{0.0, 1.0\}$ | High satisfaction heuristic indicator: $\mathbb{I}(R_p \ge 4.5)$ | Candidate specific | **Proxied** (catalog product rating $\ge 4.5$) |
| **34** | `high_discount_boost` | $\{0.0, 1.0\}$ | Bargain seeker heuristic indicator: $\mathbb{I}(\text{discount} \ge 30\%)$ | Candidate specific | **Shared Identically** (catalog product discount) |
| **35** | `intercept_bias` | $1.0$ | Constant bias term representing baseline preference expectation | $1.0$ | **Shared Identically** (constant 1.0) |

---

## 3. Historical Replay Guarantees & Equivalence

1. **Exact Mathematical Parity**:
   Offline training data generation in Phase 9B imports and calls `build_context_from_history` and `extract_feature_vector` directly from `app.services.feature_engineering`. There is no duplicate or approximation feature code.
2. **Strict Chronological Past**:
   For any interaction at timestamp $t$, only interactions strictly occurring prior to $t$ ($\tau < t$) are passed into `build_context_from_history`. The current target event is never included in the context vector.
3. **Canonical Hash Assertions**:
   Any test or training script asserts that `FEATURE_SPEC_HASH == "75f60ed70b2a"` before constructing or storing $A_{\text{hist}}$ and $b_{\text{hist}}$ matrices.

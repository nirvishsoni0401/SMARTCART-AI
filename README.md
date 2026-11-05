# SmartCart AI — Adaptive E-Commerce Platform with Contextual Multi-Armed Bandits

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF.svg?logo=vite&logoColor=white)](https://vitejs.dev)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.0-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com)
[![SQLite](https://img.shields.io/badge/SQLite-WAL--Mode-003B57.svg?logo=sqlite&logoColor=white)](https://sqlite.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Pytest-56%20Passed-brightgreen.svg)](backend/tests)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

SmartCart AI is a fully tested, verified, multi-category e-commerce web application featuring an adaptive recommendation engine powered by **Contextual Multi-Armed Bandits (LinUCB with Sherman-Morrison rank-1 updates)**. Designed to mirror enterprise e-commerce platforms like Amazon and Flipkart, it operates with Indian Rupee (₹) currency and supports six core retail departments: **Electronics, Fashion, Home & Kitchen, Beauty, Sports & Fitness, and Books** (262 active catalog SKUs: 157 DummyJSON + 105 seeded Books).

The platform tackles fundamental challenges in modern recommender systems—**exploration vs. exploitation, the cold-start dilemma, feedback loops, and dynamic preference drift**—by learning online from both positive and negative customer interactions in sub-millisecond latency ($<0.3\text{ ms}$).

---

## Table of Contents

1. [Overview & Problem Statement](#1-overview--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [36-Dimensional Context Feature Vector](#3-36-dimensional-context-feature-vector)
4. [Bandit Recommendation Algorithms](#4-bandit-recommendation-algorithms)
5. [Reward Function & Delayed Feedback Handling](#5-reward-function--delayed-feedback-handling)
6. [Quick Start & Running Instructions](#6-quick-start--running-instructions)
7. [Comprehensive API Reference](#7-comprehensive-api-reference)
8. [Admin Dashboard & Simulation Suite](#8-admin-dashboard--simulation-suite)
9. [Empirical Benchmark Results & Offline Evaluation](#9-empirical-benchmark-results--offline-evaluation)
10. [Assumptions & System Limitations](#10-assumptions--system-limitations)
11. [Future Roadmap](#11-future-roadmap)

---

## 1. Overview & Problem Statement

### The Dilemma in Modern E-Commerce Recommendations

Traditional recommendation engines rely heavily on **Collaborative Filtering (Matrix Factorization)** and **Deep Neural Networks (Two-Tower models)** trained in offline batch pipelines. While effective for static catalogs, these architectures suffer from fundamental limitations in dynamic retail environments:

1. **Feedback Loops & Popularity Traps**: Popular products receive impressions, which generate clicks, which in turn feed training data that tells the model to recommend those products even more. High-margin or emerging products are starved of traffic.
2. **Exploration vs. Exploitation Trade-off**: Pure exploitation recommends items with high historical conversion rates, maximizing immediate revenue but preventing the discovery of evolving customer preferences. Pure exploration introduces irrelevant products that alienate shoppers.
3. **The Cold-Start Challenge**: New visitors have no historical profile, and newly cataloged items have zero ratings. Collaborative filtering collapses when interaction matrices are sparse.
4. **Preference Drift**: Shoppers' preferences change over time (e.g., seasonal holidays, life stage changes, technology upgrades). Offline models cannot adapt until the next retraining cycle.

### The SmartCart AI Solution

SmartCart AI formulates recommendation as an **online Contextual Multi-Armed Bandit (MAB)** problem:

```
At each interaction round t:
1. Context Extraction: The system observes user context and candidate product features ϕ(u, p) ∈ ℝ³⁶.
2. Candidate Generation: 50 candidate products are selected using category balance and inventory rules.
3. Policy Scoring: The bandit policy (e.g. LinUCB) computes expected payoff plus an upper confidence bound.
4. Diversity & MMR Re-Ranking: Maximal Marginal Relevance limits category dominance (max 3 per department).
5. Explainability: Transparent feature attributions (θᵢ · ϕᵢ) generate human-readable recommendation reasons.
6. Real-Time Feedback Loop: User actions (click, add-to-cart, purchase, dislike, return) emit scalar rewards.
7. Online Update: LinUCB parameter vectors are updated via O(d²) Sherman-Morrison rank-1 matrix updates.
```

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer (React 19 + Vite 8 + Tailwind CSS 4)"]
        UI["Customer Storefront (:5173)\n[Home, Catalog, PDP, Cart, Wishlist, Orders]"]
        ADMIN["Admin Dashboard (:5173/admin)\n[Live Recharts, Traffic Simulator, Policy Switcher]"]
        CONSENT["Privacy Consent Banner\n[Opt-in / Opt-out Tracking]"]
    end

    subgraph API ["Application Layer (FastAPI 0.115+ :8000)"]
        AUTH["Auth Router\n[JWT + Bcrypt]"]
        CATALOG["Product Router\n[Filters, Search, Sort]"]
        ORDER["Order & Cart Router\n[Checkout, Stock Decr]"]
        RECS["Recommendation Router\n[/recommendations]"]
        EVENTS["Event & Feedback Router\n[/events, /events/impressions]"]
        ADMIN_API["Admin Management Router\n[/admin/metrics, /admin/simulate]"]
    end

    subgraph RecPipeline ["3-Stage Recommendation Pipeline"]
        STAGE1["Stage 1: Candidate Generation\n(N=50, Out-of-Stock Filter, Dept Balance)"]
        STAGE2["Stage 2: Policy Scoring\n(LinUCB, LinTS, Discounted LinUCB, ε-Greedy)"]
        STAGE3["Stage 3: Diversity & MMR\n(Max 3/dept in Top-10, Similarity Penalty)"]
        EXPLAIN["Explainability Engine\n(Top Feature Contribution θᵢ · ϕᵢ)"]
    end

    subgraph BanditEngine ["Adaptive Online Learning Core"]
        SM_UPDATE["Sherman-Morrison Rank-1 Inverse Update\nA⁻¹ ← A⁻¹ - (A⁻¹xxᵀA⁻¹) / (1 + xᵀA⁻¹x)"]
        DELAYED["Delayed Return Compensator\n(r = -18.0 to cancel +10 purchase & apply -8 return)"]
        LAZY_SKIP["Lazy Skip Resolver\n(Resolves unclicked impressions after session expiry)"]
    end

    subgraph Persistence ["Persistence Layer (SQLite WAL Mode)"]
        DB[("smartcart.db\nUsers, Products, Orders, Cart, Events")]
        MODEL_STORE[("model_states\nMatrix A⁻¹, Vector b, Theta θ")]
    end

    UI -->|HTTP / Axios| API
    ADMIN -->|Live Polling 3-5s| ADMIN_API
    CONSENT -->|Consent Toggle| AUTH

    RECS --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 --> STAGE3
    STAGE3 --> EXPLAIN
    EXPLAIN --> RECS

    EVENTS --> SM_UPDATE
    ORDER --> DELAYED
    EVENTS --> LAZY_SKIP
    SM_UPDATE --> MODEL_STORE
    MODEL_STORE --> STAGE2

    API --> DB
```

---

## 3. 36-Dimensional Context Feature Vector

Every recommendation decision evaluates a joint user-item context vector $\phi(u, p) \in \mathbb{R}^{36}$. The feature vector incorporates customer preference decays, product attributes, cross-term interactions, and temporal cyclicality.

| Index | Feature Name | Data Type | Range | Description & Rationale |
| :---: | :--- | :---: | :---: | :--- |
| **0** | `user_aff_electronics` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Electronics ($\lambda = 0.05/\text{day}$). |
| **1** | `user_aff_fashion` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Fashion products. |
| **2** | `user_aff_home_kitchen` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Home & Kitchen products. |
| **3** | `user_aff_beauty` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Beauty & Personal Care. |
| **4** | `user_aff_sports` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Sports & Fitness equipment. |
| **5** | `user_aff_books` | Float | $[0.0, 1.0]$ | Exponentially time-decayed interaction weight with Books & Publications. |
| **6** | `prod_cat_electronics` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Electronics department. |
| **7** | `prod_cat_fashion` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Fashion department. |
| **8** | `prod_cat_home_kitchen` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Home & Kitchen department. |
| **9** | `prod_cat_beauty` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Beauty department. |
| **10** | `prod_cat_sports` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Sports department. |
| **11** | `prod_cat_books` | Binary | $\{0, 1\}$ | One-hot indicator: Product is in Books department. |
| **12** | `price_ratio` | Float | $[0.0, 2.0]$ | Ratio of product price to department maximum price: $\min(2.0, p / p_{\text{max\_dept}})$. |
| **13** | `discount_pct_norm` | Float | $[0.0, 1.0]$ | Normalized promotional discount percentage: $\text{discount} / 100.0$. |
| **14** | `rating_norm` | Float | $[0.0, 1.0]$ | Normalized customer rating: $\text{rating} / 5.0$. |
| **15** | `log_review_count` | Float | $[0.0, 1.0]$ | Log-transformed social proof: $\ln(1 + \text{reviews}) / \ln(10001)$. |
| **16** | `popularity_score` | Float | $[0.0, 1.0]$ | Normalised catalogue-wide sales popularity rank. |
| **17** | `in_stock_factor` | Float | $[0.0, 1.0]$ | Logistic stock factor: $\min(1.0, \text{stock} / 20.0)$. |
| **18** | `user_price_sensitivity`| Float | $[0.0, 1.0]$ | Customer's historical mean purchase price relative to catalog range. |
| **19** | `user_discount_pref` | Float | $[0.0, 1.0]$ | Customer's historical affinity for discounted purchases. |
| **20** | `user_return_tendency` | Float | $[0.0, 1.0]$ | Fraction of customer's historical orders that resulted in a return. |
| **21** | `category_match` | Float | $[0.0, 1.0]$ | Direct dot product of user affinity vector and item category one-hot vector. |
| **22** | `price_affinity_match` | Float | $[0.0, 1.0]$ | Price compatibility score: $1.0 - |\text{price\_ratio} - \text{user\_price\_sensitivity}|$. |
| **23** | `rating_credibility` | Float | $[0.0, 1.0]$ | Composite rating reliability: $\text{rating\_norm} \times \text{log\_review\_count}$. |
| **24** | `day_of_week_sin` | Float | $[-1.0, 1.0]$ | Cyclical temporal feature: $\sin(2\pi \cdot \text{weekday} / 7)$. |
| **25** | `day_of_week_cos` | Float | $[-1.0, 1.0]$ | Cyclical temporal feature: $\cos(2\pi \cdot \text{weekday} / 7)$. |
| **26** | `hour_of_day_sin` | Float | $[-1.0, 1.0]$ | Cyclical temporal feature: $\sin(2\pi \cdot \text{hour} / 24)$. |
| **27** | `hour_of_day_cos` | Float | $[-1.0, 1.0]$ | Cyclical temporal feature: $\cos(2\pi \cdot \text{hour} / 24)$. |
| **28** | `is_weekend` | Binary | $\{0, 1\}$ | Binary weekend indicator: $1$ if Saturday or Sunday, else $0$. |
| **29** | `interact_electronics` | Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_electronics` $\times$ `prod_cat_electronics`. |
| **30** | `interact_fashion` | Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_fashion` $\times$ `prod_cat_fashion`. |
| **31** | `interact_home_kitchen`| Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_home_kitchen` $\times$ `prod_cat_home_kitchen`. |
| **32** | `interact_beauty` | Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_beauty` $\times$ `prod_cat_beauty`. |
| **33** | `interact_sports` | Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_sports` $\times$ `prod_cat_sports`. |
| **34** | `interact_books` | Float | $[0.0, 1.0]$ | Second-order interaction: `user_aff_books` $\times$ `prod_cat_books`. |
| **35** | `bias_intercept` | Constant | $1.0$ | Constant unit bias term for linear regression. |

---

## 4. Bandit Recommendation Algorithms

SmartCart AI includes six fully implemented, production-ready recommendation policies conforming to a unified `BaseRecommender` interface.

### 4.1 LinUCB with Sherman-Morrison Rank-1 Inversion

The primary algorithm is **Linear Upper Confidence Bound (LinUCB)** with disjoint parameter estimation.

#### Payoff Estimation & Upper Confidence Bound
For a context vector $x \in \mathbb{R}^d$ ($d = 36$):
$$\hat{\theta} = A^{-1} b$$
$$\text{Score}(x) = x^T \hat{\theta} + \alpha \sqrt{x^T A^{-1} x}$$

- $x^T \hat{\theta}$ represents the **expected reward** (exploitation).
- $\alpha \sqrt{x^T A^{-1} x}$ represents the **uncertainty confidence bound** (exploration).
- $\alpha \in \mathbb{R}^+$ is the exploration hyperparameter (default $\alpha = 0.5$).

#### $O(d^2)$ Sherman-Morrison Online Update
Standard ridge regression requires inverting $A \in \mathbb{R}^{d \times d}$ with complexity $O(d^3)$. In high-throughput e-commerce, this causes unacceptable latency. SmartCart AI maintains the inverse matrix $M = A^{-1}$ directly using the **Sherman-Morrison formula**:

$$M_{t+1} = M_t - \frac{M_t x_t x_t^T M_t}{1 + x_t^T M_t x_t}$$
$$b_{t+1} = b_t + r_t x_t$$

This reduces online updates to $O(d^2)$ matrix-vector multiplications, resulting in **$<0.30\text{ ms}$ update latency**.

---

### 4.2 Discounted LinUCB (Drift-Aware)

When consumer tastes shift abruptly (e.g., fashion seasonal turnover or life events), standard LinUCB accumulates past observations indefinitely, slowing down adaptation. **Discounted LinUCB** introduces an exponential forgetting factor $\gamma \in (0, 1]$:

$$A_{t+1} = \gamma A_t + (1 - \gamma) I_d + x_t x_t^T$$
$$b_{t+1} = \gamma b_t + r_t x_t$$

Using $\gamma = 0.95$, past interactions decay exponentially, granting higher agility to recent trends while maintaining stability via the $(1 - \gamma) I_d$ regularizer.

---

### 4.3 Linear Thompson Sampling (LinTS)

Rather than adding an explicit upper confidence bound, **Linear Thompson Sampling** maintains a Bayesian posterior distribution over parameters:
$$\theta \sim \mathcal{N}\left(\hat{\theta}, v^2 A^{-1}\right)$$
$$\text{Score}(x) = x^T \tilde{\theta} \quad \text{where } \tilde{\theta} \sim \mathcal{N}\left(A^{-1} b, v^2 A^{-1}\right)$$

Sampling is performed efficiently via Cholesky decomposition: $\tilde{\theta} = \hat{\theta} + v L z$ where $A^{-1} = L L^T$ and $z \sim \mathcal{N}(0, I_d)$.

---

### 4.4 Epsilon-Greedy Bandit

A classic heuristic policy that balances exploration and exploitation:
- With probability $1 - \epsilon$: Select candidate with highest predicted reward $x^T \hat{\theta}$.
- With probability $\epsilon = 0.10$: Select a random product uniformly from the candidate pool.

---

### 4.5 Popularity Prior Baseline

A non-personalized heuristic policy reflecting standard baseline e-commerce behavior:
$$\text{Score}(p) = 0.6 \times \text{popularity\_score}(p) + 0.4 \times \left(\frac{\text{rating}(p)}{5.0}\right)$$

When users decline privacy tracking consent, SmartCart AI strictly falls back to this baseline policy.

---

### 4.6 Uniform Random Policy

Selects items uniformly at random from available inventory. Serves as a theoretical lower bound for empirical regret calculation.

---

## 5. Reward Function & Delayed Feedback Handling

### 5.1 Reward Schedule

Every user interaction translates directly into a signed scalar feedback signal $r \in \mathbb{R}$:

| Action | Reward ($r$) | Type | Rationale |
| :--- | :---: | :---: | :--- |
| **Purchase** | `+10.0` | Positive | Highest commercial conversion intent; confirmed transaction. |
| **Add to Cart** | `+5.0` | Positive | Strong high-intent signal to purchase. |
| **Wishlist** | `+3.0` | Positive | Mid-funnel interest and future consideration. |
| **Product Click** | `+2.0` | Positive | Exploration of product details page. |
| **Impression / Skip** | `0.0` | Neutral | Passive view without user engagement. |
| **Remove from Cart** | `-2.0` | Negative | Hesitation, price objection, or buyer reconsideration. |
| **Dislike (Thumbs Down)** | `-5.0` | Negative | Explicit negative feedback on recommended product card. |
| **Delayed Return** | `-8.0` | Negative | Dissatisfaction post-purchase; severe supply/fit penalty. |

### 5.2 Delayed Reward Compensation Architecture

In real-world e-commerce, returns occur days or weeks after purchase. If a model was awarded $+10.0$ at purchase time, simply applying $-8.0$ later would still leave a net positive $+2.0$ reward for a returned item.

SmartCart AI solves this via **Delayed Differential Compensation**:
$$r_{\text{compensation}} = -18.0 \quad (-(10.0) + (-8.0))$$
$$b \leftarrow b + r_{\text{compensation}} \cdot x$$

This cancels out the original $+10.0$ reward and applies the full $-8.0$ penalty to the model parameter vector, immediately reducing the exposure of return-prone items.

---

## 6. Quick Start & Running Instructions

### Prerequisites
- **Python**: Version 3.11 or higher
- **Node.js**: Version 18 or higher (with npm)
- **Git**

### Installation

1. Clone or navigate to the project directory:
   ```bash
   cd c:\Users\ASUS TUF\Desktop\SMARTCART-AI
   ```

2. Configure environment variables:
   ```bash
   copy .env.example .env
   ```

3. Initialize Backend Virtual Environment & Dependencies:
   ```bash
   python -m venv backend/venv
   backend\venv\Scripts\pip install -r backend/requirements.txt
   ```

4. Install Frontend Dependencies:
   ```bash
   npm --prefix frontend install
   ```

5. Seed Database (Generates 630 realistic products across 6 departments, demo users, and admin):
   ```bash
   backend\venv\Scripts\python backend/app/seed.py
   ```

---

### Running the Platform

#### Single-Command Launcher
Run both FastAPI backend (:8000) and Vite frontend (:5173) simultaneously:
```bash
npm start
```
*(Alternative on Windows: double-click or run `run.bat`)*

#### Running Services Separately
- **FastAPI Backend**:
  ```bash
  cd backend
  ..\backend\venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
  ```
- **Vite React Frontend**:
  ```bash
  cd frontend
  npm run dev
  ```

---

### Running Tests & Experiment Benchmarks

- **Execute Backend Pytest Suite (15/15 tests)**:
  ```bash
  backend\venv\Scripts\pytest backend/tests -v
  ```

- **Run Multi-Cohort 1500-Round Simulation Experiments**:
  ```bash
  backend\venv\Scripts\python experiments/run_experiments.py
  ```
  *(Generates 7 high-resolution publication charts in `experiments/plots/` and results in `experiments/results/`)*

- **Run Offline Replay / Inverse Propensity Scoring (IPS) Evaluation**:
  ```bash
  backend\venv\Scripts\python experiments/offline_eval.py
  ```

---

### Demo Credentials

| Role | Email | Password | Access Level |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin@smartcart.ai` | `admin123` | Full access to `/admin` dashboard & model switcher |
| **Customer 1** | `rahul@smartcart.ai` | `user123` | Tech & Gadget enthusiast profile |
| **Customer 2** | `priya@smartcart.ai` | `user123` | Fashion & Beauty lover profile |
| **Customer 3** | `amit@smartcart.ai` | `user123` | Home & Kitchen shopper profile |

---

## 7. Comprehensive API Reference

All backend endpoints are served under `http://localhost:8000`. Interactive OpenAPI documentation is accessible at `http://localhost:8000/docs`.

### Authentication & Consent (`/auth`)
- `POST /auth/register` — Register a new customer account.
- `POST /auth/login` — Authenticate and receive a JWT Bearer token.
- `GET /auth/me` — Retrieve the authenticated profile.
- `PUT /auth/consent` — Toggle tracking consent. Declining immediately switches the user to the privacy-preserving popularity baseline.

### Product Catalog (`/products`)
- `GET /products` — Paginated catalog with query search (`?q=`), category filtering (`?category=`), brand, price bounds (`?min_price=`, `?max_price=`), min rating, and sorting.
- `GET /products/{id}` — Full product specifications and stock status.
- `GET /products/{id}/similar` — Attribute-based similarity recommendations.
- `GET /products/categories` — Distinct list of active departments.
- `GET /products/brands` — Distinct list of catalog brands.

### Cart & Wishlist (`/cart`, `/wishlist`)
- `GET /cart` — View current cart items, quantities, and subtotal.
- `POST /cart` — Add item to cart (automatically emits `+5.0` reward).
- `PUT /cart/items/{id}` — Update item quantity.
- `DELETE /cart/items/{id}` — Remove item from cart (automatically emits `-2.0` reward).
- `GET /wishlist` — Retrieve customer wishlist items.
- `POST /wishlist` — Add item to wishlist (`+3.0` reward).
- `DELETE /wishlist/items/{id}` — Remove item from wishlist.

### Orders & Checkout (`/orders`)
- `POST /orders` — Create order from current cart (atomically decrements inventory, creates order records, and emits `+10.0` purchase reward).
- `GET /orders` — Retrieve past customer orders.
- `POST /orders/{id}/items/{item_id}/return` — Process return request (restores stock and executes `-18.0` delayed compensation update).

### Recommendation Engine (`/recommendations`)
- `GET /recommendations`
  - Query parameters: `limit` (int, default 10), `category` (optional filter), `max_price` (optional filter), `session_id`.
  - Headers: `Authorization: Bearer <token>` (optional).
  - Response:
    ```json
    {
      "recommendations": [
        {
          "product": {
            "id": 14,
            "name": "OnePlus 12 5G (Silky Black)",
            "category": "Electronics",
            "price": 64999.0,
            "rating": 4.6
          },
          "score": 1.482,
          "ucb_bonus": 0.312,
          "was_exploration": false,
          "explanation": "High affinity for Electronics (42% match)",
          "algorithm": "linucb",
          "log_id": 104
        }
      ],
      "algorithm": "linucb",
      "cold_start": false,
      "latency_ms": 0.28
    }
    ```

### Feedback & Events (`/events`)
- `POST /events` — Record explicit customer feedback (`click`, `dislike`, `cart_add`, etc.).
- `POST /events/impressions` — Batch-record passive impressions for unclicked tracking.

### Admin Operations (`/admin`)
- `GET /admin/metrics` — Aggregate system counters, average reward, CTR, conversion rate, and active algorithm.
- `POST /admin/algorithms` — Hot-swap the active global bandit policy.
- `PUT /admin/reward-config` — Dynamically tune scalar reward weights.
- `POST /admin/simulate` — Execute synthetic multi-cohort traffic runs.
- `POST /admin/reset-models` — Reset bandit covariance matrices to identity $I_d$.
- `GET /admin/logs` — Paginated audit log of every recommendation scoring event.

---

## 8. Admin Dashboard & Simulation Suite

The Admin Dashboard is accessible at `http://localhost:5173/admin` (requires admin credentials).

```
+-----------------------------------------------------------------------------------------+
| SMARTCART AI — ADAPTIVE EXPERIMENT & CONTROL PLANE                                      |
+-----------------------------------------------------------------------------------------+
| [ Total Users: 12 ] [ Catalog: 630 ] [ Avg Reward: +4.82 ] [ Rec Latency: 0.28 ms ]     |
+-----------------------------------------------------------------------------------------+
| ACTIVE ALGORITHM SWITCHER                                                               |
| (o) LinUCB (Sherman-Morrison)   ( ) Discounted LinUCB   ( ) Linear Thompson Sampling   |
| ( ) Epsilon-Greedy (eps=0.1)     ( ) Popularity Baseline ( ) Uniform Random Policy      |
+-----------------------------------------------------------------------------------------+
| RECHARTS LIVE VISUALIZATION (Polling every 3s)                                         |
|  - Cumulative Reward Trajectories across Algorithms                                     |
|  - Conversion Funnel (Impression -> Click -> Cart -> Purchase)                         |
|  - Department Distribution Pie Chart & Diversity Radar                                 |
+-----------------------------------------------------------------------------------------+
| TRAFFIC SIMULATOR                                                                       |
| Cohorts: [Tech Enthusiast, Fashion Diva, Home Decorator, Bargain Hunter, etc.]          |
| Rounds: [ 50 ]  [ 200 ]  [ 500 ]  [ 1000 ]   [ RUN SIMULATION BATCH ]                  |
+-----------------------------------------------------------------------------------------+
| REWARD WEIGHT TUNER                                                                     |
| Purchase: [+10.0]  Add to Cart: [+5.0]  Click: [+2.0]  Dislike: [-5.0]  Return: [-8.0] |
+-----------------------------------------------------------------------------------------+
```

---

## 9. Empirical Benchmark Results & Frozen Offline Evaluation

SmartCart AI evaluates its adaptive recommendation engine through two complementary methodologies:
1. **Online Synthetic Simulation Suite**: 5 random seeds, 1,500 rounds per algorithm, measuring cumulative reward, regret, and preference drift adaptation.
2. **Hardened Proxy Offline Evaluation on Retailrocket Historical Data**: Evaluated on all **335,612 chronological test interactions** across **194,155 unique visitors** ($d=36$ feature space, `FEATURE_SPEC_HASH = 75f60ed70b2a`, frozen configuration $N_{\text{cap}} = 10,000, m = 3, \lambda = 1.0, \alpha = 0.5$).

> [!NOTE]
> **Evaluation Protocol & Framing**:
> Historical evaluations are strictly **proxy offline evaluations on logged positive-only interactions** (views, cart adds, transactions). They measure ranking performance under observed user trajectories, not counterfactual revenue. Under no circumstances is the system claimed to be "production-ready" or "causal". The system is tested and verified across 64 automated tests.
> 
> **Status**: **PERMANENTLY FROZEN (Phase 10B)** — No further evaluation phases.

---

### 9.1 Multi-Armed Bandit Online Simulation Summary

| Algorithm | Cumulative Reward | Cumulative Regret | CTR (%) | Conv. Rate (%) | NDCG@10 | Intra-List Div. | Catalog Cov. (%) | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinUCB (Sherman-Morrison)** | **6,590.8 ± 152.4** | **3,931.2 ± 109.8** | **84.19%** | **34.26%** | **0.6816** | 0.8811 | 61.14% | **0.30 ms** |
| **Discounted LinUCB ($\gamma=0.95$)** | 6,171.0 ± 118.4 | 4,195.7 ± 73.9 | 81.87% | 32.23% | 0.6735 | 0.8560 | **99.87%** | **0.27 ms** |
| **Lin Thompson Sampling (LinTS)** | 6,587.6 ± 326.2 | 3,953.9 ± 160.0 | **84.33%** | **34.53%** | **0.6863** | 0.8617 | 61.90% | 1.27 ms |
| **Epsilon-Greedy ($\epsilon=0.10$)** | 6,564.8 ± 167.1 | 3,956.0 ± 104.4 | 83.80% | 34.37% | 0.6815 | 0.8301 | 69.68% | 0.14 ms |
| **Popularity Baseline** | 6,033.8 ± 149.8 | 4,227.3 ± 88.9 | 82.32% | 30.06% | 0.6757 | 0.9153 | 31.87% | 0.12 ms |
| **Uniform Random Policy** | 5,537.0 ± 37.3 | 4,599.9 ± 15.6 | 78.29% | 29.04% | 0.6629 | **0.9553** | 100.00% | 0.10 ms |

---

### 9.2 Primary Headline: Evaluation on NOVEL Targets (Discovery First)

In e-commerce recommendation, discovery performance on **unviewed products** is the primary benchmark. Of the $335,612$ sequential test interactions, **$263,849$ events ($78.62\%$)** are novel targets where the customer has never previously interacted with the catalog product.

Following the Phase 10B frozen evaluation protocol, both **Visitor-Level (Macro)** and **Event-Level (Micro)** estimands are reported alongside **2,000-resample visitor-clustered 95% bootstrap confidence intervals**.

#### Full Catalog (262 SKUs) Ranking — Novel Targets Only ($n = 263,849$)

| Policy | Macro HR@10 (Headline) | 95% Bootstrap CI | Micro HR@10 | Paired Macro Diff vs Pop | p-value | Significance |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Popularity Baseline** | **5.76%** | $[5.66\%, 5.85\%]$ | 5.86% | Reference | — | Reference |
| **SameBrandPopularity** | 5.86% | $[5.76\%, 5.96\%]$ | 6.14% | $+0.10\%$ | $0.082$ | Negligible ($< 0.5\%$) |
| **RecentlyViewed** | 5.69% | $[5.59\%, 5.78\%]$ | 5.55% | $-0.07\%$ | $0.854$ | Negligible ($< 0.5\%$) |
| **Warm LinUCB ($\alpha=0.5$)** | 5.58% | $[5.48\%, 5.67\%]$ | 5.88% | $-0.18\%$ | $0.999$ | Negligible ($< 0.5\%$) |
| **Cold LinUCB ($\alpha=0.5$)** | 5.57% | $[5.48\%, 5.67\%]$ | 5.65% | $-0.19\%$ | $0.998$ | Negligible ($< 0.5\%$) |
| **Warm LinTS ($v=0.25$)** | 5.39% | $[5.30\%, 5.48\%]$ | 5.72% | $-0.36\%$ | $1.000$ | Negligible ($< 0.5\%$) |
| **Aggregated Warm LinUCB** | 3.11% | $[3.04\%, 3.18\%]$ | 3.74% | $-2.65\%$ | $< 0.001$ | Statistically Inferior |

> [!IMPORTANT]
> **Key Finding on Novel Discovery**:
> On novel target interactions ($78.62\%$ of events), **Warm LinUCB achieves 5.58% Macro HR@10 vs Popularity Baseline 5.76%** (paired difference $-0.18\%$, $p = 0.999$, negligible size). The contextual bandit provides **zero discovery advantage** over a static popularity baseline when recommending unviewed products.

---

### 9.3 Repeat Targets Breakdown: The Recency Cache Dominance

On the remaining **$71,763$ events ($21.38\%$)**, the user interacts with an item previously seen in their session or history:

| Policy | Macro HR@10 | 95% Bootstrap CI | Micro HR@10 | Advantage over Warm LinUCB |
| :--- | :---: | :---: | :---: | :---: |
| **RecentlyViewed** (LRU cache) | **73.22%** | $[72.77\%, 73.66\%]$ | **68.57%** | **$+20.44\%$** |
| **Warm LinUCB ($\alpha=0.5$)** | 52.78% | $[52.26\%, 53.29\%]$ | 46.04% | Reference |
| **Warm LinTS ($v=0.25$)** | 52.45% | $[51.94\%, 52.95\%]$ | 45.76% | $-0.33\%$ |
| **Aggregated Warm LinUCB** | 50.44% | $[49.91\%, 50.96\%]$ | 44.03% | $-2.34\%$ |
| **SameBrandPopularity** | 50.33% | $[49.80\%, 50.86\%]$ | 44.49% | $-2.45\%$ |
| **DepartmentRecency** | 16.79% | $[16.40\%, 17.20\%]$ | 16.47% | $-35.99\%$ |
| **Popularity Baseline** | 5.99% | $[5.72\%, 6.25\%]$ | 6.16% | $-46.79\%$ |
| **Cold LinUCB ($\alpha=0.5$)** | 5.85% | $[5.60\%, 6.10\%]$ | 5.98% | $-46.93\%$ |

*Takeaway*: While Warm LinUCB appears strong on repeats ($52.78\%$), it is vastly outperformed by a simple `RecentlyViewed` recency heuristic ($73.22\%$, $+20.44\%$ higher). The contextual bandit cannot be justified as an optimal mechanism for repeat item recommendations when an LRU cache trivially outperforms it.

---

### 9.4 Overall Full Catalog, Hybrid Slates & Candidate Exclusion v2

#### Overall Full Catalog (262 SKUs, $N = 335,612$, $194,155$ Visitors)

| Policy | Setting | Macro HR@10 | 95% Bootstrap CI | Micro HR@10 | Paired Diff vs LinUCB |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Department Heuristic** | Full Catalog | **90.25%** | $[90.16\%, 90.34\%]$ | **74.25%** | $+80.96\%$ ($p < 0.0005$) |
| **RecentlyViewed** | Full Catalog | 10.90% | $[10.79\%, 11.01\%]$ | 19.02% | $+1.61\%$ ($p < 0.0005$) |
| **Hybrid Slate (k=3 RV + 7 LinUCB)** | Full Catalog | **10.75%** | $[10.64\%, 10.85\%]$ | **18.55%** | **$+1.46\%$** ($p < 0.0005$) |
| **Hybrid Slate (k=5 RV + 5 LinUCB)** | Full Catalog | 10.64% | $[10.53\%, 10.75\%]$ | 18.78% | $+1.35\%$ ($p < 0.0005$) |
| **Warm LinTS ($v=0.25$)** | Full Catalog | 9.46% | $[9.36\%, 9.57\%]$ | 14.51% | $+0.17\%$ ($p = 0.0410$) |
| **Warm LinUCB ($\alpha=0.5$)** | Full Catalog | 9.29% | $[9.18\%, 9.40\%]$ | 14.44% | Reference |
| **Popularity Baseline** | Full Catalog | 5.76% | $[5.66\%, 5.85\%]$ | 5.92% | $-3.53\%$ ($p = 1.0000$) |
| **Cold LinUCB ($\alpha=0.5$)** | Full Catalog | 5.58% | $[5.49\%, 5.67\%]$ | 5.72% | $-3.71\%$ ($p = 1.0000$) |

*Hybrid Finding*: In overall unconstrained traffic, interleaving the top 3 items from `RecentlyViewed` with 7 slots from `Warm LinUCB` yields **10.75% Macro HR@10**, outperforming pure Warm LinUCB by **$+1.46\%$ ($p < 0.0005$)**, proving that contextual bandits benefit significantly when coupled with explicit recency caches.

#### Candidate Exclusion v2 (All Engaged Masked — Pure Discovery Task)

When **all previously engaged items are masked** from candidate sets to enforce genuine discovery:
- **Overall Stream ($N=335,612$)**: Warm LinUCB achieves **9.46% [9.35%, 9.56%]** vs Popularity **5.77% [5.67%, 5.87%]**.
- **Novel-Only Target Stream ($n=263,849$)**: Warm LinUCB achieves **5.56% [5.47%, 5.66%]** vs Popularity **5.77% [5.67%, 5.87%]** (paired diff $-0.21\%$, $p = 0.9985$, negligible). Warm LinTS achieves **5.77%** (identical to Popularity, $p=0.4940$).

---

### 9.5 Repaired Prequential Online Replay (20,000 Steps)

The anomalous Phase 10A rolling hit rates ($66\%$ to $92\%$) were traced to evaluation artifacts (inverted candidate subset scoring). Repaired chronological prequential replay over 20,000 steps demonstrates realistic trajectories ($w=1,000$ rolling window):

| Step | Cold LinUCB Rolling (w=1000) | Warm LinUCB Rolling (w=1000) | Cold Cumulative HR@10 | Warm Cumulative HR@10 | Novel-Only Warm Cumulative |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 0.00% | 10.00% | 0.00% | 10.00% | 8.51% |
| **500** | 1.20% | 13.60% | 1.20% | 13.60% | 6.18% |
| **2,000** | 3.70% | 13.80% | 2.75% | 13.70% | 5.79% |
| **5,000** | 12.40% | 14.00% | 8.02% | 13.96% | 5.87% |
| **7,200 (Catch-Up)** | **14.60%** | **14.60%** | 9.42% | 14.12% | 5.82% |
| **10,000** | 14.30% | 13.50% | 11.00% | 14.15% | 5.79% |
| **15,000** | 14.20% | 15.90% | 12.29% | 15.07% | 5.85% |
| **20,000** | 17.70% | 17.40% | **12.96%** | **14.73%** | **5.88%** |

> [!WARNING]
> **Prequential Catch-Up vs Cumulative Lead**:
> In **rolling terms** ($w=1,000$), Cold LinUCB catches up with Warm LinUCB at **Step 7,200** ($14.60\%$ vs $14.60\%$) as it adapts online. However, in **cumulative terms**, Warm LinUCB retains its exploration lead across all 20,000 steps ($14.73\%$ vs $12.96\%$). Cold LinUCB cumulative HR@10 never surpasses Warm LinUCB within 20,000 steps; Claim C7 remains formally withdrawn.
| **20,000** | 17.70% | 17.40% | **12.96%** | **14.73%** | **5.88%** |

> [!WARNING]
> **Prequential Crossover Claim Withdrawn**:
> Cold LinUCB cumulative HR@10 **never surpasses Warm LinUCB** within 20,000 test steps (Cold reaches $12.96\%$ vs Warm $14.73\%$). While Cold's rolling window approaches Warm due to repeat interaction learning, cumulative performance remains lower. Phase 10A Claim C7 is formally withdrawn.

---

### 9.6 Valid Leave-Group-Out Ridge Refit Ablations

Previous ablations zeroed feature columns at test time without refitting, causing dot-product mismatch. Phase 10B performs mathematically valid **leave-group-out Ridge refits** using sliced sufficient statistics $(A_{\text{sub}} + \lambda I)^{-1} b_{\text{sub}}$ evaluated across all $335,612$ test events:

| Ablation Configuration | Dim $d$ | Overall Macro HR@10 | Novel Macro HR@10 | Repeat Macro HR@10 | Relative Repeat Impact |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Features (Full Model)** | 36 | 9.37% | 5.58% | 52.78% | Baseline |
| **Ablate: Category Affinities** | 30 | 9.15% | 5.37% | 52.84% | $+0.1\%$ |
| **Ablate: Brand Affinity (Feature 6)** | 35 | **5.63%** | **5.63%** | **5.74%** | **$-89.1\%$ (Collapse)** |
| **Ablate: Price & Budget Group** | 32 | 9.36% | 5.56% | 53.00% | $+0.4\%$ |
| **Ablate: Product Category One-Hot** | 30 | 9.56% | 5.92% | 51.37% | $-2.7\%$ |
| **Ablate: Price, Rating, Discount** | 33 | 9.38% | 5.57% | 52.89% | $+0.2\%$ |
| **Ablate: Product Popularity** | 35 | 9.34% | 5.57% | 52.71% | $-0.1\%$ |
| **Ablate: Brand Hash Buckets** | 28 | 9.00% | 5.21% | 52.64% | $-0.3\%$ |
| **Ablate: Session & Intent Features** | 33 | 9.37% | 5.58% | 52.78% | $0.0\%$ |
| **Ablate: High Boosts & New Flag** | 33 | 9.37% | 5.57% | 52.79% | $0.0\%$ |

*Mechanistic Explanation*: Feature 6 (`user_brand_affinity`) is non-zero for **100.0% of repeat events** (mean $0.9177$), but only **5.72% of novel events** (mean $0.0465$). The Ridge solver assigns $\theta_6 = 1.28$, turning Feature 6 into a massive repeat-intent booster (+1.18 score boost). Slicing out Feature 6 collapses repeat HR from $52.78\%$ to $5.74\%$ while novel HR is completely untouched ($5.58\% \rightarrow 5.63\%$).

---

### 9.7 Cold-Start Shrinkage Hypothesis Confirmation

- **Hypothesis**: The Cold LinUCB score ($\alpha \|\phi\|_2$) acts as an implicit popularity prior because higher-rated, discounted catalog products have larger feature norms.
- **Empirical Result**: Spearman rank correlation between cold-start feature norms and catalog popularity priors is **$r = 0.7126$ ($p = 6.59 \times 10^{-42}$)**, confirming the hypothesis.

---

### 9.8 Generated Publication Research Figures

All Phase 10B figures are generated at 300 DPI in `experiments/plots/retailrocket/`:
1. `novel_first_headline_10b.png`: Primary headline comparison on novel targets with visitor-clustered 95% error bars.
2. `repeat_vs_novel_baselines_10b.png`: Disaggregated breakdown showing `RecentlyViewed` dominating repeats.
3. `prequential_hitrate_10b.png`: Repaired 20,000-step prequential learning curves (no crossover).
4. `valid_ablations_refit_10b.png`: Leave-group-out Ridge refit waterfall across feature groups.
5. `cold_start_shrinkage_correlation_10b.png`: Scatter and trend confirming $r = 0.7126$ popularity prior correlation.
6. `candidate_exclusion_v2_discovery_10b.png`: Pure discovery task performance under full engagement masking.

---

## 10. Assumptions & System Limitations

1. **Proxy Offline Evaluation vs Unbiased Counterfactual Estimation**:
   Historical evaluation uses logged positive interactions (`view`, `addtocart`, `transaction`). Because historical exploration propensities were unlogged, counterfactual inverse propensity scoring (IPS) cannot be computed on this split without unprovable assumptions. Results measure proxy ranking performance under observed user trajectories.
2. **Behavioral Proxy Mapping**:
   Retailrocket product IDs are anonymized item identifiers mapped via category and price proxies to the 262-SKU catalog. Results demonstrate transferability of contextual behavior (brand affinity, price sensitivity, category preference), rather than item-level identity.
3. **One-Step Contextual Bandit vs MDP**:
   Recommendations are framed as immediate one-step contextual bandit decisions rather than a multi-step Markov Decision Process (MDP). While optimal for sub-millisecond ranking, it does not explicitly optimize multi-month customer lifetime value (LTV).
4. **Database Concurrency in SQLite**:
   SQLite with Write-Ahead Logging (WAL) handles high-concurrency read queries cleanly, but matrix updates serialize writes at the database level. For deployments exceeding 10,000 writes/second, Redis or PostgreSQL is recommended.
5. **Candidate Retrieval Floor**:
   Candidate retrieval evaluates 50 candidate items per request. For extreme catalog scales ($>1,000,000$ SKUs), Approximate Nearest Neighbor (ANN) indexing (HNSW, ScaNN) would be required.

---

## 11. Future Roadmap

- [ ] **Deep Neural Contextual Bandits**: Integration of Neural-Linear bandits with pretrained CLIP / BERT embeddings for rich visual and textual catalog representations.
- [ ] **Cross-Session Graph Embeddings**: Graph Neural Networks (GNNs) capturing co-purchase topology across millions of customer graphs.
- [ ] **Distributed Streaming Ingestion**: Apache Kafka integration for decoupled event emission and asynchronous bandit model updates across multi-region edge clusters.
- [ ] **A/B Testing Harness**: Multi-armed bandit traffic routing (Thompson Sampling meta-policy) to dynamically allocate incoming traffic to superior recommendation algorithms.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

# SmartCart AI — 10-Minute Live Demonstration Script & Oral Limitations

This document provides a minute-by-minute demonstration walkthrough of SmartCart AI for technical evaluators, stakeholders, and reviewers, followed by the mandatory list of epistemic limitations to state aloud.

---

## 1. Quick Environment Verification (Pre-Flight)

Before beginning the demo, verify that both services are running:
```bash
# Terminal 1: Backend API (:8000)
.\backend\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Storefront Frontend (:5173)
cd frontend && npm run dev
```
Open your browser to `http://localhost:5173`.

---

## 2. 10-Minute Live Demonstration Flow

### Minute 0:00 – 1:30 | Registration & Privacy Consent Flow
1. **Navigate to Storefront**: Open `http://localhost:5173`.
2. **Register New Shopper**:
   - Click **Register / Login** in the navigation header.
   - Create a new account (e.g., `aarti@example.com` / `Shopper123`).
   - Highlight the **Privacy Consent Banner**: note that SmartCart AI requires explicit opt-in for personal contextual bandit tracking. Confirm that tracking consent is checked.
3. **Initial Cold Start State**:
   - Observe the initial "Recommended For You" carousel.
   - Explain: *"As a brand-new user with zero interaction history, the system seamlessly falls back to our Department Heuristic and catalog popularity prior. No user data is hallucinated."*

---

### Minute 1:30 – 3:30 | Category Browsing & High-Intent Action
1. **Browse Electronics**:
   - Navigate to the **Electronics** category from the navigation bar or category pills.
   - Click on 2–3 products (e.g., *Apple iPhone 15 Pro*, *Sony WH-1000XM5 Headphones*).
   - Point out product detail pages (PDP) with high-resolution imagery, INR (₹) pricing, and specs.
2. **Add to Cart**:
   - Add the *Sony Headphones* to the shopping cart.
   - Explain: *"This action emits a `cart_add` interaction event, yielding a high-intent $+5.0$ reward directly to our online bandit model."*

---

### Minute 3:30 – 5:30 | Online Adaptation & Transparent Explanations
1. **Return to Home Page**:
   - Click the **SmartCart AI** logo to return to the home screen.
2. **Observe Real-Time Adaptation**:
   - The "Recommended For You" row has adapted immediately without page reload or batch retraining.
   - Electronics SKUs now dominate the top slots.
3. **Inspect Explainability Pills**:
   - Point to the explanation badge beneath the recommendation title:
     `Because you browse Electronics • High Category Affinity (92%)`
   - Point to an exploratory slot:
     `Exploration Pick • Active Cross-Department Discovery`
   - Explain: *"Our Two-Stage Hybrid (Dept-Restricted LinUCB) reserves exploration slots across the catalog to prevent popularity traps and category echo chambers."*
4. **Inspect "Recently Viewed" Row**:
   - Scroll down to the dedicated **Recently Viewed** row.
   - Note the exact items just visited.
   - Explain: *"The Recently Viewed row is powered strictly by the customer's personal browsing session. It is completely decoupled from the bandit models and does NOT distort exploration-exploitation reward matrices."*

---

### Minute 5:30 – 6:30 | Explicit Negative Feedback Handling
1. **Dislike an Unwanted Recommendation**:
   - Locate a recommended product card (e.g., in a secondary department).
   - Click the **Thumbs Down / Dislike** icon.
2. **Observe Instant Feedback Update**:
   - The card animates with a feedback toast (`Disliked: Feedback recorded`).
   - Explain: *"Unlike conventional recommender systems that only track positive clicks, SmartCart AI treats dislikes as a $-5.0$ penalty in the online ridge regression model, instantly depressing feature weights for that product profile."*

---

### Minute 6:30 – 8:30 | Admin Dashboard & Real-Time Telemetry
1. **Open Admin Dashboard**:
   - Navigate to `http://localhost:5173/admin` (or click **Admin** in the header).
   - Log in using administrative credentials (`admin@smartcart.ai` / `admin123`).
2. **Examine Live System Telemetry**:
   - **Active Catalog**: 262 SKUs (157 DummyJSON + 105 seeded Books).
   - **Latency Percentiles**: Point to the sub-millisecond bandit scoring latency ($<0.30\text{ ms}$ p50) enabled by Sherman-Morrison $O(d^2)$ rank-1 matrix inversion.
   - **Origin Filtering**: Toggle between `live`, `simulated`, and `all` data origins. Explain how simulated data is isolated from production metrics.
3. **Snapshot & Versioning Controls**:
   - Demonstrate the **Snapshot** button: archives an immutable snapshot of all active model weight matrices into `model_states`.
   - Demonstrate the **Rollback** button: restores previous weights in the event of an operational anomaly.

---

### Minute 8:30 – 9:30 | Policy Switching & Robustness Help Panel
1. **Inspect Policy Robustness Panel**:
   - Scroll to the **Policy Robustness Across Synthetic Environments** panel.
   - Point out the empirical regret share table from our 30-seed simulator benchmark:
     - Well-Specified Base: Dept-Restricted LinUCB achieves **17.9% regret share** (outperforming Department Heuristic at 28.4%).
     - Nonlinear Interactions: Department Heuristic collapses to **71.8% regret share**, while adaptive bandits (Discounted LinUCB at 26.6% and Cold LinUCB at 30.6%) maintain superior robustness.
2. **Switch Active Policy**:
   - In the **Active Recommendation Policy** switcher, click **Discounted LinUCB** or **Dept Heuristic**.
   - Note the instant confirmation toast and active algorithm status change.
3. **Run Multi-Cohort Simulation**:
   - Set **Simulate Traffic** to 25 rounds in **Sandbox Mode (sim:*)**.
   - Click **Run Simulation**: watch simulated events populate without touching live production model weights.

---

### Minute 9:30 – 10:00 | Oral Limitations & Wrap-Up
Read aloud the mandatory limitations listed in Section 3 below.

---

## 3. Mandatory List of Limitations to State Aloud

When presenting SmartCart AI, evaluators must read or state the following five limitations aloud:

1. **One-Step RL Scope**:
   > *"Contextual bandits are myopic, one-step reinforcement learning agents. They optimize immediate click and purchase payoffs per slate interaction, but do not model multi-step sequential state transitions or long-term customer lifetime value (LTV)."*

2. **Synthetic Simulator Assumptions**:
   > *"All simulation benchmark results describe algorithm behaviour strictly under the calibrated simulator's stated synthetic assumptions—specifically our 10 user archetypes, position examination decay $\gamma_k = 1/\log_2(k+2)$, and logistic utility models. They must never be cited as causal evidence of real-world customer lift."*

3. **Offline Retailrocket Study as Proxy**:
   > *"The Retailrocket offline evaluation was permanently frozen in Phase 10D as a proxy evaluation on logged positive-only interactions. Because historical logs lack unobserved impression logs, offline replay metrics are subject to exposure selection bias."*

4. **Warm-Start Status in Production**:
   > *"Retailrocket warm-start snapshots remain strictly disabled in our live models. While moderate warm-start ($N_{\text{cap}}=500$) helped in synthetic simulation, mapping third-party historical categories to a live catalog risks negative transfer bias under domain shift."*

5. **No Universal Dominance**:
   > *"No single algorithm dominates across all retail environments. While hybrid candidate restriction excels under strong category clustering, pure adaptive bandits dominate under complex cross-feature interactions. Exposing policy selection to administrators is a deliberate architectural necessity."*

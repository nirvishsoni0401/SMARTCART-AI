# SmartCart AI — Calibrated Simulator v2 Assumption Card

> [!IMPORTANT]
> **Epistemic Scope & Boundaries**:
> All results produced by this simulator describe model behaviour **strictly under the stated synthetic assumptions**.
> They do **not** represent production readiness, do **not** establish causal customer impact in the wild,
> and must **never** be cited as evidence of real-world shopper behaviour without an online A/B test.

---

## 1. Environment & Persona Architecture
- **Catalog Ground Truth**: 262 active products across 6 departments (Beauty: 13, Books: 105, Electronics: 38, Fashion: 49, Home & Kitchen: 40, Sports: 17).
- **Latent Personas (10 Archetypes)**:
  1. *Tech Gadgeteer*: Heavy Electronics affinity (0.75), Apple/Samsung/Sony preference, ₹38,000 budget.
  2. *Fashion Trendsetter*: Heavy Fashion affinity (0.80), luxury brands, ₹45,000 budget, high discount sensitivity.
  3. *Avid Bookworm*: Heavy Books affinity (0.85), publishing houses preference, ₹850 budget.
  4. *Home Organizer*: Heavy Home & Kitchen affinity (0.75), IKEA/Philips, ₹14,000 budget.
  5. *Active Athlete*: Heavy Sports affinity (0.75), Nike/Adidas/Puma, ₹3,200 budget.
  6. *Clean Beauty Lover*: Heavy Beauty affinity (0.80), Nykaa/Forest Essentials, ₹3,500 budget.
  7. *Budget Bargain Hunter*: Multi-category discount seeker, ₹1,200 budget, 0.95 discount sensitivity.
  8. *Luxury Connoisseur*: High-end Electronics & Fashion, ₹120,000 budget, price-insensitive.
  9. *Student / Tech Learner*: Technical Books & Electronics, ₹700 budget.
  10. *Casual Family Shopper*: Multi-department family essentials, ₹4,500 budget.
- **Arrival Mixture**: 25% new user cold-start arrivals, 75% returning users with persistent interaction histories.

---

## 2. Slate Presentation & Examination Bias
- **Slate Size**: $K = 10$ items presented simultaneously.
- **Position Examination Model**:
  $$\gamma_k = \\frac{1}{\\log_2(k + 2)} \\quad \\text{for } k \\in \\{0, \\dots, 9\\}$$
  Strictly monotonic: $\\gamma_0 = 1.000 > \\gamma_1 = 0.631 > \\dots > \\gamma_9 = 0.289$.

---

## 3. Latent Utility & Conditional Funnel Calibration
- **Utility Formulation**:
  $$z(u, i) = w_{\\text{cat}} \\cdot \\text{aff}(u, c_i) + w_{\\text{brand}} \\cdot \\mathbb{I}(b_i \\in \\mathcal{B}_u) + w_{\\text{rating}} \\cdot \\frac{r_i}{5} - w_{\\text{price}} \\cdot \\text{pen}(p_i, B_u) + w_{\\text{disc}} \\cdot \\text{disc}_i + u_{\\text{repeat}} - \\theta_{\\text{click}} + \\epsilon$$
  Where:
  - $w_{\\text{cat}} = 2.2, w_{\\text{brand}} = 0.8, w_{\\text{rating}} = 0.6, w_{\\text{price}} = 1.0, w_{\\text{disc}} = 0.5$
  - $\\theta_{\\text{click}} = 5.72$ (calibrated offset)
  - Difficulty noise: Easy $\\sigma = 0.15$, Medium $\\sigma = 0.40$, Hard $\\sigma = 0.80$.
- **Empirical Anchors (Retailrocket Calibration)**:
  - $P(\\text{view} \\to \\text{cart}) \\approx 2.6\\%$
  - $P(\\text{cart} \\to \\text{transaction}) \\approx 32.4\\%$
  - Random Slate CTR $\\approx 3 - 6\\%$
  - Oracle Slate CTR $\\approx 20 - 35\\%$

---

## 4. Central Reward Specification
Verified against `backend/app/config.py`:
- Purchase: **+10.0**
- Cart Add: **+5.0**
- Wishlist: **+3.0**
- Click: **+2.0**
- Dislike: **-5.0**
- Cart Removal: **-2.0**
- Delayed Return: **-8.0**
- Skip / No interaction: **0.0**

---

## 5. Non-Stationary Drift & Cold Inventory
- **Abrupt Drift**: At $t = 10,000$ (50% horizon), persona category affinities cyclically shift and budgets expand by 30%.
- **Gradual Drift**: Continuous linear drift from $t = 5,000$ to $t = 15,000$.
- **Cold-Start Item Arrivals**: 12 products (2 from each department) withheld until $t = 10,000$, testing exploratory discovery.

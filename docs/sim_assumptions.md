# SmartCart AI — Calibrated Simulator v2 Assumption Card

> [!IMPORTANT]
> **Epistemic Scope & Boundaries**:
> All results produced by this simulator describe model behaviour **strictly under the stated synthetic assumptions**.
> They do **not** represent production readiness, do **not** establish causal customer impact in the wild,
> and must **never** be cited as evidence of real-world shopper behaviour without an online A/B test.

---

## 1. Environment & Persona Architecture
- **Catalog Ground Truth**: 262 active products across 6 departments (Beauty: 13, Books: 105, Electronics: 38, Fashion: 49, Home & Kitchen: 40, Sports: 17).
- **User Population**:
  - Initial active pool: 150 users, expanding to 1,000 unique synthetic users across 20,000 rounds.
  - User arrival process: Poisson-distributed sessions with mean 5.0 sessions per active customer.
  - Returning shoppers: 75% sampled from active pool according to power-law engagement.
  - History persistence: FIFO session buffer (capacity 20) with lifetime preservation of clicks, carts, purchases, and wishlists.
- **Latent Personas (10 Archetypes)**:
  1. *Tech Gadgeteer*: Heavy Electronics affinity (0.75), Apple/Samsung/Sony, ₹38,000 budget.
  2. *Fashion Trendsetter*: Heavy Fashion affinity (0.80), luxury brands, ₹45,000 budget, high discount sensitivity.
  3. *Avid Bookworm*: Heavy Books affinity (0.85), publishing houses, ₹850 budget.
  4. *Home & Kitchen Organizer*: Heavy Home & Kitchen affinity (0.75), IKEA/Philips, ₹14,000 budget.
  5. *Active Athlete*: Heavy Sports affinity (0.75), Nike/Adidas/Puma, ₹3,200 budget.
  6. *Clean Beauty Lover*: Heavy Beauty affinity (0.80), Nykaa/Forest Essentials, ₹3,500 budget.
  7. *Budget Bargain Hunter*: Multi-category discount seeker, ₹1,200 budget, 0.95 discount sensitivity.
  8. *Luxury Connoisseur*: High-end Electronics & Fashion, ₹120,000 budget, price-insensitive.
  9. *Student / Tech Learner*: Technical Books & Electronics, ₹700 budget.
  10. *Casual Family Shopper*: Multi-department family essentials, ₹4,500 budget.

---

## 2. Slate Presentation & Examination Bias
- **Slate Size**: $K = 10$ items presented simultaneously without replacement.
- **Position Examination Model**:
  $$\gamma_k = \frac{1}{\log_2(k + 2)} \quad \text{for } k \in \{0, \dots, 9\}$$
  Strictly monotonic: $\gamma_0 = 1.000 > \gamma_1 = 0.631 > \dots > \gamma_9 = 0.289$.

---

## 3. Latent Utility & Conditional Funnel Calibration
- **Utility Formulation (Exact Code Implementation in `experiments/sim_v2/environment.py`)**:
  $$u(u, i) = w_{\text{cat}} \text{aff}_{\text{cat}} + w_{\text{brand}} \mathbb{I}_{\text{brand}} + w_{\text{rating}} r_i - w_{\text{price}} \text{excess}_i + w_{\text{disc}} d_i + \text{bonus}_{\text{rep}} - \beta_{\text{click}} + \epsilon$$
  where $\beta_{\text{click}} = 5.72$ is the baseline logit threshold (renamed from $\theta$ to prevent notation clash with LinUCB), and $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
- **Canonical Utility Parameter Weights**:
  - $w_{\text{cat}} = 2.2$ (Category affinity dominates decision-making)
  - $w_{\text{brand}} = 0.8$ (Brand affinity match)
  - $w_{\text{rating}} = 0.6$ (Catalog rating fit)
  - $w_{\text{price}} = 1.0$ (Budget excess penalty)
  - $w_{\text{disc}} = 0.5$ (Discount sensitivity incentive)
  - $\beta_{\text{click}} = 5.72$ (Logit offset)
- **Historical Parameter Evolution Note**:
  In Phase 11, the initial specification proposed weights (2.2, 1.0, 0.6). During calibration against empirical Retailrocket funnel anchors, weights were tuned to anchor $P(\text{click})$, $P(\text{cart} \mid \text{click})$, and $P(\text{tx} \mid \text{cart})$. In Phase 12A, documentation inadvertently printed typographic approximations (2.4, 0.8, 0.7, 0.5, 0.4); Phase 12A-2 aligns all documentation with the exact canonical values implemented in code (`w_cat=2.2, w_brand=0.8, w_rating=0.6, w_price=1.0, w_disc=0.5, beta_click=5.72`).
- **Repeat Engagement Bonus**:
  $$\text{bonus}_{\text{rep}} = 0.75 \cdot \exp(-0.0005 \Delta t)$$
  accumulates specifically on items previously presented to and engaged with by the user.
- **Funnel Stage Multipliers**:
  - $P(\text{cart} \mid \text{click}) = 2.6\% \times \text{clip}(0.5 + 0.5 \cdot \text{aff}_{\text{cat}} + 0.3 \cdot \mathbb{I}_{\text{in-budget}} - 0.3 \cdot \text{excess}, 0.5, 2.0)$
  - $P(\text{tx} \mid \text{cart}) = 32.4\% \times \text{clip}(0.6 + 0.4 \cdot r_i + 0.2 \cdot d_i, 0.6, 1.5)$
- **Non-Cart Events & Reward Values**:
  - Purchase: $+10$ points
  - Cart Add: $+5$ points
  - Wishlist ($p = 1.5\%$): $+3$ points
  - Click: $+2$ points
  - Dislike ($p = 1.0\%$): $-5$ points
  - Cart Removal ($p = 5.0\%$): $-2$ points
  - Delayed Return ($p = 3.0\%$): $-8$ points
  - Skip: $0$ points

---

## 4. Calibration Parameters & Difficulty Noise Scales

| Difficulty | Noise $\sigma$ | Diagnostic Random CTR (%) | Diagnostic Oracle CTR (%) | $P(\text{cart} \mid \text{click})$ (%) | $P(\text{tx} \mid \text{cart})$ (%) | Carts (Sample Size) | Purchases |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Easy** | 0.15 | 5.47% | 20.08% | 2.66% | 43.33% | 30 | 13 |
| **Medium** | 0.40 | 5.42% | 21.43% | 2.43% | 34.48% | 29 | 10 |
| **Hard** | 0.80 | 7.05% | 25.75% | 2.47% | 27.03% | 37 | 10 |

### Calibration Reconciliations & Benchmark CTR Parity:
1. **Diagnostic vs Headline Horizon Saturation**:
   The diagnostic calibration table above measures pre-drift initial saturation over a short horizon ($T = 4,000$ rounds, seed 42) without preference drift, yielding Medium Oracle CTR of 21.43% and Random CTR of 5.42%.
   In contrast, the headline benchmark is evaluated across the full $T = 20,000$ rounds over 30 independent seeds with preference drift at $t=10,000$, yielding Medium Oracle Slate CTR of 17.39% and Random Slate CTR of 4.97%.
2. **Sample Sizes Behind Conversion Funnel**:
   Under medium difficulty calibration, 1193 clicks yielded 29 cart adds ($P(\text{cart} \mid \text{click}) = 2.43\%$), which subsequently converted into 10 purchases ($P(\text{tx} \mid \text{cart}) = 34.48\%$).
3. **Random and Oracle Slate CTR by Environment Variant (30 Evaluation Seeds, Horizon 20,000)**:
   - **Well-Specified Base**: Random CTR = 4.97% | Oracle CTR = 17.39%
   - **Hidden Style Factor**: Random CTR = 6.05% | Oracle CTR = 31.63%
   - **Nonlinear Interactions**: Random CTR = 6.96% | Oracle CTR = 47.84%
   - **High Funnel Conversion**: Random CTR = 4.98% | Oracle CTR = 17.38%

### Jensen's Inequality Convexity Effect:
On negative logit ranges (where $\beta_{\text{click}} = 5.72$ places latent utility in the negative domain), the sigmoid activation $\sigma(x) = 1 / (1 + e^{-x})$ is strictly convex. By Jensen's inequality:
$$\mathbb{E}[\sigma(X + \epsilon)] > \sigma(\mathbb{E}[X])$$
Consequently, higher Gaussian logit noise ($\sigma = 0.80$ at Hard) elevates empirical click probability (7.05%) compared to low noise ($\sigma = 0.15$ at Easy: 5.47%). This convexity asymmetry is a natural mathematical feature of logistic choice models under additive Gaussian noise.

### Empirical Reward Composition (% of absolute points generated):
- **Easy Difficulty**: Click: 75.8%, Cart: 5.0%, Purchase: 4.4%, Wishlist: 6.7%, Dislike: 8.1%, Return: 0.0%, Cart Remove: 0.0%.
- **Medium Difficulty**: Click: 76.4%, Cart: 4.6%, Purchase: 3.2%, Wishlist: 7.5%, Dislike: 8.3%, Return: 0.0%, Cart Remove: 0.0%.
- **Hard Difficulty**: Click: 78.1%, Cart: 4.8%, Purchase: 2.6%, Wishlist: 6.1%, Dislike: 8.1%, Return: 0.2%, Cart Remove: 0.1%.

---

## 5. Model Misspecification Variants & Realisability
The benchmark tests 3 deliberate violations of LinUCB's 36-dimensional linear assumption:
1. **Hidden Style Factor**: Unobserved 3-dimensional latent style embedding ($w_{\text{style}} = 1.2$) interacting with user style preference; absent from feature vector $\phi(u, i)$.
2. **Nonlinear Interactions**: Price $\times$ category sensitivity offsets and rating $\times$ discount synergies.
3. **High Funnel Conversion**: Base $P(\text{cart} \mid \text{click}) = 8.0\%$ and $P(\text{tx} \mid \text{cart}) = 65.0\%$, making downstream conversion dominate click feedback.

---

## 6. Mathematical Identities & Trajectory Oracle
For any policy $\pi$, step regret is defined as the gap between the optimal oracle slate reward along policy $\pi$'s own trace and the expected reward of the served slate:
$$\text{step\_regret}_t = R^*(u_t, t) - R(u_t, a_t)$$
Summing over all $T$ rounds yields the exact mathematical identity:
$$\text{Cumulative Reward}(\pi) + \text{Cumulative Regret}(\pi) \equiv R^*_{\text{trajectory}}(\pi) = \sum_{t=1}^T R^*(u_t, t)$$
Because user repeat bonuses accumulate specifically along the active items served, policies serving higher-utility items build greater repeat affinity, raising the trajectory oracle ceiling (8,132.7 for Oracle, 7,855.4 for Cold LinUCB, and 7,598.1 for Popularity).

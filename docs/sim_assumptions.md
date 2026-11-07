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
  - Initial active pool: 150 users.
  - New arrival probability: 25% per round ($p_{\text{new}} = 0.25$).
  - Returning shoppers: 75% sampled from active pool according to power-law engagement.
  - History persistence: Full lifetime history of clicks, carts, purchases, and wishlists preserved across sessions.
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
- **Slate Size**: $K = 10$ items presented simultaneously.
- **Position Examination Model**:
  $$\gamma_k = \frac{1}{\log_2(k + 2)} \quad \text{for } k \in \{0, \dots, 9\}$$
  Strictly monotonic: $\gamma_0 = 1.000 > \gamma_1 = 0.631 > \dots > \gamma_9 = 0.289$.

---

## 3. Latent Utility & Conditional Funnel Calibration
- **Utility Formulation**:
  $$u(u, i) = w_{\text{cat}} \text{aff}_{\text{cat}} + w_{\text{brand}} \mathbb{I}_{\text{brand}} + w_{\text{rating}} r_i - w_{\text{price}} \text{excess}_i + w_{\text{disc}} d_i + \text{bonus}_{\text{rep}} - \beta_{\text{click}} + \epsilon$$
  where $\beta_{\text{click}} = 5.72$ is the baseline logit threshold (renamed from $\theta$ to prevent notation clash with LinUCB), and $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
- **Repeat Engagement Bonus**:
  $$\text{bonus}_{\text{rep}} = 0.75 \cdot \exp(-0.0005 \Delta t)$$
  added for products previously viewed by the shopper.
- **Funnel Stage Multipliers**:
  - $P(\text{cart} \mid \text{click}) = 2.6\% \times \text{clip}(0.5 + 0.5 \cdot \text{aff}_{\text{cat}} + 0.3 \cdot \mathbb{I}_{\text{in-budget}} - 0.3 \cdot \text{excess}, 0.5, 2.0)$
  - $P(\text{tx} \mid \text{cart}) = 32.4\% \times \text{clip}(0.6 + 0.4 \cdot r_i + 0.2 \cdot d_i, 0.6, 1.5)$
- **Non-Cart Events**:
  - Wishlist: $6.0\%$ ($R = +3$)
  - Dislike: $4.0\%$ ($R = -5$)
  - Cart Removal (unpurchased): $5.0\%$ ($R = -2$)
  - Delayed Return (purchased): $3.5\%$ ($R = -8$)

---

## 4. Calibration Parameters & Difficulty Noise Scales

| Difficulty | Noise $\sigma$ | Random Slate CTR (%) | Oracle Slate CTR (%) | $P(\text{cart} \mid \text{click})$ (%) | $P(\text{tx} \mid \text{cart})$ (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Easy** | 0.15 | 5.23% | 19.93% | 0.63% | 0.00% |
| **Medium** | 0.40 | 5.43% | 22.00% | 1.21% | 0.00% |
| **Hard** | 0.80 | 6.97% | 25.87% | 1.85% | 0.00% |

### Empirical Reward Composition (% of absolute points generated):
- **Medium Difficulty**: Click: 79.9%, Cart Add: 2.4%, Purchase: 0.0%, Wishlist: 8.0%, Dislike: 9.7%, Return: 0.0%, Cart Remove: 0.0%.

---

## 5. Model Misspecification Variants & Realisability
The benchmark tests 3 deliberate violations of LinUCB's 36-dimensional linear assumption:
1. **Hidden Style Factor**: Unobserved 3-dimensional latent style embedding ($w_{\text{style}} = 1.2$) interacting with user style preference; absent from $\phi(u, i)$.
2. **Nonlinear Interactions**: Price $\times$ category sensitivity offsets and rating $\times$ discount synergies.
3. **High Funnel Conversion**: Base $P(\text{cart} \mid \text{click}) = 8.0\%$ and $P(\text{tx} \mid \text{cart}) = 65.0\%$, making downstream conversion dominate click feedback.

---

## 6. Oracle Regret Resolution
In Phase 11, the sequential execution loop shared the mutable `UserInstance` object across all policies within round $t$. By the time `OraclePolicy` selected its slate, earlier policies had mutated `user.history` with fresh click events, causing `get_oracle_slate()` (invoked without cached expected values) to select items against updated repeat bonuses, while `env.step()` evaluated regret against pre-round cached expected values. This subtle state drift accumulated to 159.5 regret over 20,000 rounds.
In Phase 11B, each policy's trajectory is strictly isolated with its own environment under Common Random Numbers, ensuring the Oracle slate identically matches the argmax expected item values at every round, making Oracle cumulative regret strictly and provably **0.000** across all rounds and seeds.

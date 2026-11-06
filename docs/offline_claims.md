# SmartCart AI — Offline Evaluation Claims & Scientific Status

> [!IMPORTANT]
> **PERMANENT FREEZE DECLARATION**: All offline evaluation on Retailrocket proxy data is **permanently frozen**. No further offline benchmarks will be conducted. All subsequent validation moves strictly to the interactive customer simulator and live online bandit learning.

## 1. Supported Claims (With Verifiable Evidence)

1. **Recency-Bandit Hybrid Dominance**: Interleaving top recency items with contextual bandit exploration statistically significantly outperforms pure Warm LinUCB on overall traffic (13.16% vs 11.20%, paired difference +1.97% [1.88%, 2.05%], $p < 0.0005$). Tested specifically for $k=3$ and $k=5$.
2. **Prequential Learning Regret Avoidance**: Warm-start LinUCB avoids early cold-start cumulative regret. Cold LinUCB's rolling learning rate enters within one standard error of Warm LinUCB at step 8349, but Warm LinUCB preserves a persistent cumulative lead across 20,000 chronological steps.
3. **Absence of Contextual Exploration Edge on Novel Discovery**: On completely novel target items (unseen by the shopper), Warm LinUCB (5.82%) and Warm LinTS (5.75%) do not outperform a simple Popularity baseline (5.76%). In positive-only logged click data, bandit exploration cannot overcome popularity priors on pure novel discovery.
4. **Repeat Traffic Intent Dominance**: Recommending RecentlyViewed items on repeat target interactions achieves 99.81% Macro HR@10 under lifetime history and 73.23% under 30-min session history. In e-commerce session logs, short-term browsing memory dominates ranking.
5. **Discounted Horizon Optimality**: In static validation prequential streams, un-discounted LinUCB ($\gamma=1.0000$, 16.23%) outperforms discounted variants ($\gamma \in [0.99, 0.9999]$), demonstrating that historical knowledge should not be artificially decayed when preference drift is low.

## 2. Revised or Withdrawn Claims Since Phase 9B

1. **WITHDRAWN: 'Warm LinUCB vastly outperforms baselines on overall discovery'**: The 15.75% micro hit rate reported in Phase 9C was driven by repeat interactions and target session inclusion. On novel discovery, LinUCB equals Popularity.
2. **WITHDRAWN: 'No negative transfer on cold visitors'**: On visitors with 0 prior history, Cold LinUCB achieves parity or slight advantages over Warm LinUCB due to covariance shrinkage penalizing head items.
3. **WITHDRAWN: 'Department Heuristic achieves 90-97% hit rate'**: Formally identified as a double bug (target category leakage and zero-score rank-1 tie inflation). Corrected Department Heuristic achieves realistic hit rates.
4. **WITHDRAWN: 'Cross-mapping superiority across category_rank and cooccurrence_cluster'**: Formally retracted due to evaluating models across permuted representations and severe mapping concentration artifacts.

## 3. Methodological Limitations

1. **Proxy Product Mapping**: Retailrocket items were structurally mapped to DummyJSON/seed catalog SKUs using volume and co-occurrence proxies. While behavioral dynamics (session length, recency, funnels) are authentic, individual product semantics are synthetic.
2. **Positive-Only Logged Feedback**: Retailrocket logs contain exclusively positive interactions (clicks, add-to-carts, transactions). Unobserved impressions are treated as pseudo-negatives, which introduces sample selection bias.
3. **No Counterfactual Validity**: Offline replay on logged observational data cannot measure how users would have behaved had different items been recommended (no unconfounded propensity scores). True causal lift can only be evaluated in the simulator or via live A/B experimentation.
4. **Repeat Dominance**: Over 21% of test events represent repeat interactions on the same product. Recommenders that prioritize recency achieve massive hit rates on repeats, masking pure discovery dynamics.

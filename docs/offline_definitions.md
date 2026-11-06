# SmartCart AI — Offline Evaluation Definitions and Dataset Counts

> [!IMPORTANT]
> Every count $n$ (events) and $n_v$ (visitors) across the SmartCart AI offline evaluation repository is defined and governed by this document. All numbers are auto-generated directly from `experiments/results/offline_master_results.json`.

## 1. Mathematical Estimand Definitions

### 1.1 Macro Hit Rate at 10 (Macro HR@10)
Macro HR@10 measures the unweighted average recommendation quality across individual shoppers (visitors):

$$\text{Macro HR@10} = \frac{1}{|V_S|} \sum_{u \in V_S} \frac{1}{|E_u \cap S|} \sum_{e \in E_u \cap S} \mathbb{I}(\text{target}(e) \in \text{Top10}(e))$$

where:
- $S$ is the evaluated interaction slice (e.g. Overall, Novel, Repeat).
- $V_S$ is the set of visitors who have **at least one interaction in slice $S$** ($|E_u \cap S| \ge 1$). Visitors with zero interactions in slice $S$ are strictly excluded from the denominator $|V_S|$.
- $\mathbb{I}(\cdot)$ is the binary indicator indicating whether the ground-truth target item is in the top-10 ranked slate.

### 1.2 Micro Hit Rate at 10 (Micro HR@10)
Micro HR@10 measures the global event-averaged recommendation accuracy across all interactions in the slice:

$$\text{Micro HR@10} = \frac{1}{|S|} \sum_{e \in S} \mathbb{I}(\text{target}(e) \in \text{Top10}(e))$$

## 2. Interaction Slices and History Scopes

- **Novel Target**: An event where the target catalog product has never appeared in the visitor's prior lifetime test history ($	au < t$).
- **Repeat Target**: An event where the visitor has previously interacted with the target catalog product earlier in the test stream ($	au < t$). *Note*: The repeat subset is defined using the ground-truth target product, making ranking history first near-trivial in this slice.
- **Lifetime History**: The sequence of up to 20 prior interactions for the visitor, persisted across sessions throughout the entire test stream prior to the current event timestamp ($	au < t$). Reflects a production environment with authenticated accounts or persistent cookies.
- **30-Minute Session History**: The sequence of prior interactions within the active 30-minute session. History strictly resets whenever the elapsed time between consecutive events for the visitor exceeds 1,800,000 ms (30 minutes) or when a new visitor begins.
- **Candidate Exclusion v1**: Retracts all products added to cart in the active session from recommendation candidates.
- **Candidate Exclusion v2 (Pure Discovery)**: Masks all products previously engaged by the visitor in their active history from the candidate pool, except the current ground-truth target. Explicitly evaluates whether the recommender can surface novel unviewed items.
- **Cold Context**: An interaction occurring at the very start of a 30-minute session where prior within-session history is empty (length 0).
- **First-Exposure Event**: An event representing the first time a specific catalog product is interacted with within a 30-minute session.

## 3. Authoritative Dataset Counts Table

| Slice / Subset Name | Number of Events ($n$) | Number of Visitors ($n_v$) | Definition & Scope |
| :--- | :--- | :--- | :--- |
| **Total Test Stream** | 335,612 | 194,155 | Complete held-out test split (Aug 28 – Sep 18, 2015) |
| **Novel Target Events** | 263,849 | 194,155 | First time product viewed by visitor in test stream |
| **Repeat Target Events** | 71,763 | 31,609 | Product previously engaged by visitor in test stream |
| **Cold-Context Session Starts** | 226,410 | 194,155 | Events starting a 30-min session (history length 0) |
| **Warm-Context Within-Session** | 109,202 | 194,155 | Events with $\ge 1$ within-session prior interaction |
| **First-Exposure Events** | 287,176 | 194,155 | First interaction per session-product pair |
| **Lifetime Visitor First Events** | 194,155 | 194,155 | Very first test interaction for each unique visitor |

## 4. Root-Cause Explanation of Prior Discrepancies

1. **Cold-Context Count Discrepancy (226,410 vs 120,402)**: 226,410 is the verified count of events that start a 30-minute session (history length 0). 120,402 was an erroneously hand-typed/transposed number in an earlier conversational chat summary and never existed in the data.
2. **First-Exposure Count Discrepancy (287,176 vs 194,155)**: 287,176 is the event count of session-product first exposures. 194,155 is the visitor count (or the count of each visitor's very first lifetime interaction in the test stream).
3. **Exclusion v2 Score Discrepancy (5.56% vs 5.63%)**: 5.56% is the novel-target Candidate Exclusion v2 Macro HR@10. 5.63% was the novel-target score in the Feature 6 Brand Affinity ablation table. They were conflated in a previous draft table.

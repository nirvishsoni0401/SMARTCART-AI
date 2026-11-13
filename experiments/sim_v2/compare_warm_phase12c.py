import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
before_path = ROOT / "experiments" / "results" / "sim_v2_master_results_pre12c.json"
after_path = ROOT / "experiments" / "results" / "sim_v2_master_results.json"

try:
    with open(before_path, "r", encoding="utf-8") as f:
        before = json.load(f)
except Exception:
    with open(before_path, "r", encoding="utf-16") as f:
        before = json.load(f)

with open(after_path, "r", encoding="utf-8") as f:
    after = json.load(f)

def get_val(d, k, sub="mean"):
    v = d.get(k, {})
    if isinstance(v, dict):
        return v.get(sub, 0)
    elif isinstance(v, (int, float)):
        return v
    return 0

def get_ci(d, k):
    v = d.get(k, {})
    if isinstance(v, dict):
        ci = v.get("ci_95", [0, 0])
        return ci[0], ci[1]
    return 0, 0

print("================================================================================")
print("=== PART A2: BEFORE vs AFTER REGENERATION OF WARM-START UNITS ===")
print("================================================================================\n")

print("--- 1. Headline Benchmark (N_cap=10,000, Abrupt Drift, Medium Difficulty, Base) ---")
print(f"{'Policy':<15} | {'Metric':<14} | {'Before Phase 12C':<26} | {'After Phase 12C':<26} | {'Delta':<10}")
print("-" * 98)

for pol in ["Warm LinUCB", "Warm LinTS"]:
    b_data = before["primary_benchmark"].get(pol, {})
    a_data = after["primary_benchmark"].get(pol, {})
    
    b_reg = get_val(b_data, "cumulative_regret")
    a_reg = get_val(a_data, "cumulative_regret")
    b_ci = get_ci(b_data, "cumulative_regret")
    a_ci = get_ci(a_data, "cumulative_regret")
    b_ci_str = f"[{b_ci[0]:.1f}, {b_ci[1]:.1f}]"
    a_ci_str = f"[{a_ci[0]:.1f}, {a_ci[1]:.1f}]"
    print(f"{pol:<15} | Regret         | {b_reg:7.2f} {b_ci_str:<17} | {a_reg:7.2f} {a_ci_str:<17} | {a_reg - b_reg:+8.2f}")
    
    b_rew = get_val(b_data, "cumulative_reward")
    a_rew = get_val(a_data, "cumulative_reward")
    print(f"{pol:<15} | Reward         | {b_rew:7.2f}                   | {a_rew:7.2f}                   | {a_rew - b_rew:+8.2f}")
    
    b_ctr = get_val(b_data, "slate_ctr")
    a_ctr = get_val(a_data, "slate_ctr")
    print(f"{pol:<15} | CTR            | {b_ctr:7.4f}                   | {a_ctr:7.4f}                   | {a_ctr - b_ctr:+8.4f}")
    
    b_sh = get_val(b_data, "regret_share") * 100
    a_sh = get_val(a_data, "regret_share") * 100
    print(f"{pol:<15} | Regret Share % | {b_sh:7.2f}%                  | {a_sh:7.2f}%                  | {a_sh - b_sh:+8.2f}%")
    
    # Quarterly
    b_q = b_data.get("quarter_regrets", [])
    a_q = a_data.get("quarter_regrets", [])
    for q_idx in range(min(len(b_q), len(a_q))):
        print(f"{pol:<15} | Q{q_idx+1} Regret      | {b_q[q_idx]:7.2f}                   | {a_q[q_idx]:7.2f}                   | {a_q[q_idx] - b_q[q_idx]:+8.2f}")

print("\n--- 2. Warm Sweep (N_cap in {0, 500, 2000, 10000}) ---")
print(f"{'Policy':<15} | {'N_cap':<6} | {'Before Regret':<14} | {'After Regret':<14} | {'Delta Regret':<12} | {'Before Rew':<10} | {'After Rew':<10} | {'Delta Rew':<10}")
print("-" * 102)
for pol in ["Warm LinUCB", "Warm LinTS"]:
    b_sweep = before.get("warm_start_sweep", {}).get(pol, {})
    a_sweep = after.get("warm_start_sweep", {}).get(pol, {})
    for ncap in [0, 500, 2000, 10000]:
        b_n = b_sweep.get(str(ncap), {})
        a_n = a_sweep.get(str(ncap), {})
        b_r = get_val(b_n, "cumulative_regret")
        a_r = get_val(a_n, "cumulative_regret")
        b_rw = get_val(b_n, "cumulative_reward")
        a_rw = get_val(a_n, "cumulative_reward")
        print(f"{pol:<15} | {ncap:<6} | {b_r:14.2f} | {a_r:14.2f} | {a_r - b_r:+12.2f} | {b_rw:10.2f} | {a_rw:10.2f} | {a_rw - b_rw:+10.2f}")

print("\n--- 3. Misspecification Scenarios ---")
print(f"{'Policy':<15} | {'Scenario':<25} | {'Before Regret':<14} | {'After Regret':<14} | {'Delta Regret':<12} | {'Before Share':<12} | {'After Share':<12}")
print("-" * 110)
for pol in ["Warm LinUCB", "Warm LinTS"]:
    for scen in ["hidden_style", "nonlinear_interactions", "high_funnel_conversion"]:
        b_m = before.get("misspecification_robustness", {}).get(scen, {})
        a_m = after.get("misspecification_robustness", {}).get(scen, {})
        b_s = b_m.get(pol, {})
        a_s = a_m.get(pol, {})
        b_r = get_val(b_s, "cumulative_regret")
        a_r = get_val(a_s, "cumulative_regret")
        b_sh = get_val(b_s, "regret_share") * 100
        a_sh = get_val(a_s, "regret_share") * 100
        print(f"{pol:<15} | {scen:<25} | {b_r:14.2f} | {a_r:14.2f} | {a_r - b_r:+12.2f} | {b_sh:11.2f}% | {a_sh:11.2f}%")

print("\n--- 4. Confounded Scenarios ---")
print(f"{'Policy':<15} | {'Scenario':<25} | {'Before Regret':<14} | {'After Regret':<14} | {'Delta Regret':<12}")
print("-" * 72)
for pol in ["Warm LinUCB", "Warm LinTS"]:
    for scen in ["drift_only", "new_products_only", "both"]:
        b_c = before.get("confounded_scenarios", {}).get(scen, {})
        a_c = after.get("confounded_scenarios", {}).get(scen, {})
        b_s = b_c.get(pol, {})
        a_s = a_c.get(pol, {})
        b_r = get_val(b_s, "cumulative_regret")
        a_r = get_val(a_s, "cumulative_regret")
        print(f"{pol:<15} | {scen:<25} | {b_r:14.2f} | {a_r:14.2f} | {a_r - b_r:+12.2f}")

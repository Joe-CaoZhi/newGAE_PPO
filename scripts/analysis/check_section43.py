#!/usr/bin/env python3
"""检查Section 4.3 HalfCheetah Mann-Whitney表格的数据来源"""
import json
import os

import numpy as np
from scipy import stats

BASE = "/Users/joe-caozhi/newGAE_ppo/results"

def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_final_reward(data):
    if "final_reward" in data:
        return data["final_reward"]
    if "all_eval_rewards" in data:
        r = data["all_eval_rewards"]
        return float(np.mean(r[-5:]))
    if "eval_rewards" in data:
        r = data["eval_rewards"]
        return float(np.mean(r[-5:]))
    return None

# Table in section 4.3:
# | vs. Standard PPO | 10 | 0.690 | −0.32 (small) | n.s. |
# | vs. PPO_KLPEN | 16 | 0.548 | +0.46 (small) | n.s. |
# | vs. PPO_Anneal | 10 | 0.690 | −0.38 (small) | n.s. |
# | vs. PPO_EntDecay | 13 | 1.000 | +0.07 (negligible) | n.s. |
# | vs. PPO_VClip | 6 | 0.222 | −0.98 (medium) | n.s. |
# | vs. PPO_Full | 17 | 0.421 | +0.80 (medium) | n.s. |
# Note: HCGAE (828 ± 113) performs slightly below Standard PPO (902 ± 90)

# Load UnifiedComparison data for HalfCheetah
unified_dir = os.path.join(BASE, "UnifiedComparison", "HalfCheetah-v4")
seeds = [42, 123, 456, 789, 1234]

def load_seeds(alg_dir, alg_name):
    rewards = []
    for s in seeds:
        fn = os.path.join(alg_dir, alg_name, f"{alg_name}_s{s}.json")
        if os.path.exists(fn):
            data = load_json(fn)
            r = get_final_reward(data)
            if r is not None:
                rewards.append(r)
    return rewards

algs = ["HCGAE_Imp12", "Standard_PPO", "PPO_KLPEN", "PPO_Anneal", "PPO_EntDecay", "PPO_VClip"]
data = {}
for alg in algs:
    rewards = load_seeds(unified_dir, alg)
    if rewards:
        data[alg] = rewards
        print(f"  {alg}: mean={np.mean(rewards):.0f}±{np.std(rewards, ddof=1):.0f} (n={len(rewards)})")
    else:
        print(f"  {alg}: NO DATA")

# Check BaselineComparison for PPO_Full
baseline_dir = os.path.join(BASE, "BaselineComparison", "HalfCheetah-v4")
ppo_full_rewards = load_seeds(baseline_dir, "PPO_Full_Baseline")
if ppo_full_rewards:
    data["PPO_Full_Baseline"] = ppo_full_rewards
    print(f"  PPO_Full_Baseline: mean={np.mean(ppo_full_rewards):.0f}±{np.std(ppo_full_rewards, ddof=1):.0f} (n={len(ppo_full_rewards)})")

print("\n--- Mann-Whitney comparisons (from UnifiedComparison data) ---")
if "HCGAE_Imp12" in data:
    hcgae = data["HCGAE_Imp12"]
    print(f"HCGAE_Imp12: mean={np.mean(hcgae):.0f}±{np.std(hcgae,ddof=1):.0f}")
    for alg in ["Standard_PPO", "PPO_KLPEN", "PPO_Anneal", "PPO_EntDecay", "PPO_VClip"]:
        if alg in data:
            other = data[alg]
            u_stat, p = stats.mannwhitneyu(hcgae, other, alternative='two-sided')
            pool_std = np.sqrt((np.std(hcgae,ddof=1)**2 + np.std(other,ddof=1)**2) / 2)
            d = (np.mean(hcgae) - np.mean(other)) / pool_std if pool_std > 0 else 0
            print(f"  vs {alg}: U={u_stat:.0f}, p={p:.3f}, d={d:.2f}, other_mean={np.mean(other):.0f}±{np.std(other,ddof=1):.0f}")
    if "PPO_Full_Baseline" in data:
        pf = data["PPO_Full_Baseline"]
        u_stat, p = stats.mannwhitneyu(hcgae, pf, alternative='two-sided')
        pool_std = np.sqrt((np.std(hcgae,ddof=1)**2 + np.std(pf,ddof=1)**2) / 2)
        d = (np.mean(hcgae) - np.mean(pf)) / pool_std if pool_std > 0 else 0
        print(f"  vs PPO_Full: U={u_stat:.0f}, p={p:.3f}, d={d:.2f}, PPO_Full_mean={np.mean(pf):.0f}±{np.std(pf,ddof=1):.0f}")

print("\n--- Paper section 4.3 claims: HCGAE (828 ± 113) ---")
print("This doesn't match either UnifiedComparison or ICMLExperiment data!")
print(f"UnifiedComparison HCGAE: {np.mean(data.get('HCGAE_Imp12', [0])):.0f}")
print(f"ICMLExperiment Optimal_HCGAE: 1250")
print("The 828 number appears to be from an old/different experiment run")


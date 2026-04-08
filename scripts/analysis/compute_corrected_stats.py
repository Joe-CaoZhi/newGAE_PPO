#!/usr/bin/env python3
"""
计算论文需要的所有正确统计数据
用于修正Section 4.3的HalfCheetah表格
"""
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

def load_baseline_seeds(env, alg):
    """从 BaselineComparison 加载5个种子"""
    alg_dir = os.path.join(BASE, "BaselineComparison", env, alg)
    seeds = [42, 123, 456, 789, 1234]
    rewards = []
    for s in seeds:
        # Try both naming conventions
        fn1 = os.path.join(alg_dir, f"{alg}_s{s}_metrics.json")
        fn2 = os.path.join(alg_dir, f"{alg}_s{s}.json")
        for fn in [fn1, fn2]:
            if os.path.exists(fn):
                data = load_json(fn)
                r = get_final_reward(data)
                if r is not None:
                    rewards.append(r)
                break
    return rewards

def mann_whitney_stats(a, b):
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    pool_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pool_std if pool_std > 0 else 0
    pct = (np.mean(a) - np.mean(b)) / np.mean(b) * 100
    return {
        "u": float(u), "p": float(p), "d": float(d),
        "pct": float(pct),
        "mean_a": float(np.mean(a)), "std_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)), "std_b": float(np.std(b, ddof=1)),
        "n_a": len(a), "n_b": len(b)
    }

def label_d(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"

print("=" * 70)
print("HalfCheetah BaselineComparison 数据验证")
print("=" * 70)

# Load all HalfCheetah baselines from BaselineComparison
env = "HalfCheetah-v4"
algs = ["HCGAE_Imp12", "Standard_PPO", "PPO_KLPEN", "PPO_Anneal", "PPO_EntDecay", "PPO_VClip", "PPO_Full_Baseline"]
bc_data = {}

for alg in algs:
    rewards = load_baseline_seeds(env, alg)
    if rewards:
        bc_data[alg] = rewards
        print(f"  {alg}: mean={np.mean(rewards):.0f}±{np.std(rewards,ddof=1):.0f} (n={len(rewards)})")
    else:
        print(f"  {alg}: NO DATA")

print("\n--- Computed Mann-Whitney statistics ---")
if "HCGAE_Imp12" in bc_data:
    hcgae = bc_data["HCGAE_Imp12"]
    comparisons = [
        ("Standard_PPO", "Standard PPO"),
        ("PPO_KLPEN", "PPO-KLPEN"),
        ("PPO_Anneal", "PPO-Anneal"),
        ("PPO_EntDecay", "PPO-EntDecay"),
        ("PPO_VClip", "PPO-VClip"),
        ("PPO_Full_Baseline", "PPO-Full"),
    ]

    results = []
    for alg_key, alg_label in comparisons:
        if alg_key in bc_data:
            other = bc_data[alg_key]
            r = mann_whitney_stats(hcgae, other)
            sig = "**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else "n.s.")
            d_label = label_d(r["d"])
            print(f"  HCGAE vs {alg_label}: U={r['u']:.0f}, p={r['p']:.3f}, d={r['d']:.2f} ({d_label}), {sig}")
            results.append({
                "baseline": alg_label,
                "u": r["u"], "p": r["p"], "d": r["d"],
                "d_label": d_label, "sig": sig,
                "hcgae_mean": r["mean_a"], "hcgae_std": r["std_a"],
                "other_mean": r["mean_b"], "other_std": r["std_b"]
            })

# Save for use in paper update
output = {
    "env": "HalfCheetah-v4",
    "data_source": "results/BaselineComparison/HalfCheetah-v4/",
    "hcgae_mean": float(np.mean(bc_data.get("HCGAE_Imp12", [0]))),
    "hcgae_std": float(np.std(bc_data.get("HCGAE_Imp12", [0]), ddof=1)),
    "std_ppo_mean": float(np.mean(bc_data.get("Standard_PPO", [0]))),
    "std_ppo_std": float(np.std(bc_data.get("Standard_PPO", [0]), ddof=1)),
    "comparisons": results if "HCGAE_Imp12" in bc_data else []
}
out_path = os.path.join(BASE, "baseline_halfcheetah_stats.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to: {out_path}")

# Also check: what are the actual Hopper/Walker unified comparison numbers?
print("\n" + "=" * 70)
print("Hopper/Walker BaselineComparison 数据汇总 (for reference)")
print("=" * 70)
for env in ["Hopper-v4", "Walker2d-v4"]:
    print(f"\n  {env}:")
    for alg in algs:
        rewards = load_baseline_seeds(env, alg)
        if rewards:
            print(f"    {alg}: {np.mean(rewards):.0f}±{np.std(rewards,ddof=1):.0f}")
        else:
            print(f"    {alg}: NO DATA")


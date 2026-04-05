#!/usr/bin/env python3
"""分析 AutoSCR 实验结果"""
import json
from pathlib import Path

import numpy as np

base = Path("results/AutoSCRExperiment/Hopper-v4")
results = {}

for algo in ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v2_AutoSCR"]:
    algo_dir = base / algo
    if not algo_dir.exists():
        continue
    finals = []
    for f in sorted(algo_dir.glob("*.json")):
        d = json.load(open(f))
        er = d.get("eval_rewards", [])
        if er:
            finals.append(np.mean(er[-5:]))
    if finals:
        results[algo] = {
            "mean": np.mean(finals),
            "std": np.std(finals, ddof=1) if len(finals) > 1 else 0,
            "n": len(finals),
            "seeds": finals
        }

print("=" * 60)
print("Hopper-v4 AutoSCR Experiment Results")
print("=" * 60)
for algo, data in results.items():
    print(f"\n{algo}:")
    print(f"  Mean: {data['mean']:.1f} ± {data['std']:.1f} (n={data['n']})")
    print(f"  Seeds: {[f'{x:.1f}' for x in data['seeds']]}")

# 计算关键对比
print("\n" + "=" * 60)
print("Key Comparisons")
print("=" * 60)

if "Optimal_HCGAE_v2" in results and "Optimal_PPO" in results:
    v2 = results["Optimal_HCGAE_v2"]["seeds"]
    ppo = results["Optimal_PPO"]["seeds"]
    imp = (np.mean(v2) - np.mean(ppo)) / np.mean(ppo) * 100
    pooled_std = np.sqrt((np.std(v2, ddof=1)**2 + np.std(ppo, ddof=1)**2) / 2)
    d = (np.mean(v2) - np.mean(ppo)) / (pooled_std + 1e-8)
    print(f"HCGAE v2 vs Optimal PPO: {imp:+.1f}% (d={d:.2f})")

if "Optimal_HCGAE_v2_AutoSCR" in results and "Optimal_HCGAE_v2" in results:
    auto = results["Optimal_HCGAE_v2_AutoSCR"]["seeds"]
    v2 = results["Optimal_HCGAE_v2"]["seeds"]
    imp = (np.mean(auto) - np.mean(v2)) / np.mean(v2) * 100
    pooled_std = np.sqrt((np.std(auto, ddof=1)**2 + np.std(v2, ddof=1)**2) / 2)
    d = (np.mean(auto) - np.mean(v2)) / (pooled_std + 1e-8)
    print(f"AutoSCR vs HCGAE v2: {imp:+.1f}% (d={d:.2f})")

if "Optimal_HCGAE_v2_AutoSCR" in results and "Optimal_PPO" in results:
    auto = results["Optimal_HCGAE_v2_AutoSCR"]["seeds"]
    ppo = results["Optimal_PPO"]["seeds"]
    imp = (np.mean(auto) - np.mean(ppo)) / np.mean(ppo) * 100
    pooled_std = np.sqrt((np.std(auto, ddof=1)**2 + np.std(ppo, ddof=1)**2) / 2)
    d = (np.mean(auto) - np.mean(ppo)) / (pooled_std + 1e-8)
    print(f"AutoSCR vs Optimal PPO: {imp:+.1f}% (d={d:.2f})")


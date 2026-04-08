#!/usr/bin/env python3
"""读取 ICMLExperiment/Ant-v4/Optimal_HCGAE_v3 结果（500K步）并与已有基线对比。"""
import json
from pathlib import Path

import numpy as np

icml_dir = Path("results/ICMLExperiment/Ant-v4")

algos_to_show = [
    "Standard_PPO",
    "Optimal_PPO",
    "Optimal_HCGAE",
    "Optimal_HCGAE_SCR",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v2_NoBdry",
    "Optimal_HCGAE_v2_NoGate",
    "Optimal_HCGAE_v3",
]

print("=" * 72)
print("  ICMLExperiment Ant-v4 完整结果（500K步）")
print("=" * 72)
print(f"  {'算法':<38} {'均值':>8}  {'标准差':>7}  {'种子数'}")
print("-" * 72)

results = {}
for algo in algos_to_show:
    rewards = []
    for s in range(5):
        fp = icml_dir / algo / f"{algo}_s{s}.json"
        if fp.exists():
            d = json.load(open(fp))
            rewards.append(d["final_reward"])
    if rewards:
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        results[algo] = rewards
        print(f"  {algo:<38} {mean_r:>8.1f}  {std_r:>7.1f}  [{len(rewards)}/5]")
    else:
        print(f"  {algo:<38} {'—':>8}  {'—':>7}  [0/5]")

print("=" * 72)

if "Optimal_HCGAE_v3" in results and "Optimal_PPO" in results:
    v3 = results["Optimal_HCGAE_v3"]
    ppo = results["Optimal_PPO"]
    print(f"\n  v3 vs Optimal_PPO:  {100*(np.mean(v3)-np.mean(ppo))/abs(np.mean(ppo)):+.1f}%")
    if "Optimal_HCGAE_v2" in results:
        v2 = results["Optimal_HCGAE_v2"]
        print(f"  v3 vs HCGAE_v2:     {100*(np.mean(v3)-np.mean(v2))/abs(np.mean(v2)):+.1f}%")


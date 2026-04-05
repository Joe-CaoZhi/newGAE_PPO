#!/usr/bin/env python3
"""快速读取 AntV3Validation 当前结果。"""
import json
from pathlib import Path

import numpy as np

base = Path("results/AntV3Validation")
algos = [
    "Standard_PPO", "Optimal_PPO", "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v3", "Optimal_HCGAE_v3_NoClamp",
    "Optimal_HCGAE_v3_NoVWGate", "Optimal_HCGAE_v3_NoBdryPrior",
]
seeds = [0, 1, 2]

print("=" * 72)
print("  Ant-v4 快速验证结果（200K步）")
print("=" * 72)
print(f"  {'算法':<42} {'均值':>8}  {'标准差':>7}  种子奖励")
print("-" * 72)
for algo in algos:
    rewards = []
    for s in seeds:
        fp = base / algo / f"{algo}_s{s}.json"
        if fp.exists():
            d = json.load(open(fp))
            rewards.append(round(d["final_reward"], 1))
    if rewards:
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"  {algo:<42} {mean_r:>8.1f}  {std_r:>7.1f}  {rewards}")
    else:
        print(f"  {algo:<42} {'—':>8}  {'—':>7}  [无数据]")
print("=" * 72)

# 对比摘要
v2_data, v3_data, ppo_data = [], [], []
for s in seeds:
    for name, lst in [("Optimal_HCGAE_v2", v2_data), ("Optimal_HCGAE_v3", v3_data), ("Optimal_PPO", ppo_data)]:
        fp = base / name / f"{name}_s{s}.json"
        if fp.exists():
            lst.append(json.load(open(fp))["final_reward"])

if v3_data and ppo_data:
    print(f"\n关键对比：")
    print(f"  v3 vs PPO:  {100*(np.mean(v3_data)-np.mean(ppo_data))/abs(np.mean(ppo_data)+1e-8):+.1f}%")
    if v2_data:
        print(f"  v3 vs v2:   {100*(np.mean(v3_data)-np.mean(v2_data))/abs(np.mean(v2_data)+1e-8):+.1f}%")


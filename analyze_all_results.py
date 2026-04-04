#!/usr/bin/env python3
"""
全面分析所有ICMLExperiment实验结果
汇总所有环境、所有算法的性能数据
"""
import os
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path("/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment")

def get_score(json_file):
    """从JSON文件读取最终得分（最后5次评估的均值）"""
    try:
        with open(json_file) as f:
            data = json.load(f)
        # 优先用eval_rewards
        if "eval_rewards" in data and len(data["eval_rewards"]) >= 5:
            return float(np.mean(data["eval_rewards"][-5:]))
        elif "eval_rewards" in data and len(data["eval_rewards"]) > 0:
            return float(np.mean(data["eval_rewards"]))
        elif "final_reward" in data:
            return float(data["final_reward"])
        elif "mean_reward" in data:
            return float(data["mean_reward"])
        else:
            return None
    except Exception as e:
        return None

def analyze_env_algo(env, algo):
    """分析某个环境+算法的所有seed结果"""
    algo_dir = BASE_DIR / env / algo
    if not algo_dir.exists():
        return []

    scores = []
    files = sorted(algo_dir.glob("*.json"))
    for f in files:
        s = get_score(f)
        if s is not None and isinstance(s, (int, float)):
            scores.append(float(s))

    return scores

# 所有环境和算法
envs = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4", "Ant-v4"]
algos = [
    "Standard_PPO",
    "Optimal_PPO",
    "Optimal_HCGAE",
    "Optimal_HCGAE_SCR",
    "Optimal_HCGAE_v2",
    "Optimal_HCGAE_v2_NoBdry",
    "Optimal_HCGAE_v2_NoGate",
]

print("=" * 90)
print("完整实验结果汇总")
print("=" * 90)

all_data = {}
for env in envs:
    print(f"\n{'='*30} {env} {'='*30}")
    all_data[env] = {}

    for algo in algos:
        scores = analyze_env_algo(env, algo)
        n = len(scores)
        if n == 0:
            print(f"  {algo:<35} N/A (无数据)")
            all_data[env][algo] = {"n": 0, "scores": []}
        else:
            mean = np.mean(scores)
            std = np.std(scores)
            min_v = np.min(scores)
            max_v = np.max(scores)
            per_seed = ", ".join([f"{s:.0f}" for s in scores])
            print(f"  {algo:<35} n={n}  {mean:.0f}±{std:.0f}  [{min_v:.0f}, {max_v:.0f}]")
            print(f"    per-seed: {per_seed}")
            all_data[env][algo] = {"n": n, "mean": mean, "std": std, "scores": scores}

print("\n" + "="*90)
print("消融分析：v2组成成分的重要性")
print("="*90)

for env in envs:
    print(f"\n{env}:")

    base = all_data[env].get("Optimal_PPO", {})
    v1 = all_data[env].get("Optimal_HCGAE", {})
    v2 = all_data[env].get("Optimal_HCGAE_v2", {})
    nobd = all_data[env].get("Optimal_HCGAE_v2_NoBdry", {})
    nogate = all_data[env].get("Optimal_HCGAE_v2_NoGate", {})

    base_mean = base.get("mean", None)
    v1_mean = v1.get("mean", None)
    v2_mean = v2.get("mean", None)
    nobd_mean = nobd.get("mean", None)
    nogate_mean = nogate.get("mean", None)

    if base_mean:
        print(f"  Optimal_PPO (基线):          {base_mean:.0f} ± {base.get('std', 0):.0f}  n={base.get('n', 0)}")
    if v1_mean:
        diff = (v1_mean - base_mean) / base_mean * 100 if base_mean else 0
        print(f"  Optimal_HCGAE (v1):          {v1_mean:.0f} ± {v1.get('std', 0):.0f}  n={v1.get('n', 0)}  vs Opt_PPO: {diff:+.1f}%")
    if v2_mean:
        diff_vs_base = (v2_mean - base_mean) / base_mean * 100 if base_mean else 0
        diff_vs_v1 = (v2_mean - v1_mean) / v1_mean * 100 if v1_mean else 0
        print(f"  Optimal_HCGAE_v2 (全):       {v2_mean:.0f} ± {v2.get('std', 0):.0f}  n={v2.get('n', 0)}  vs Opt_PPO: {diff_vs_base:+.1f}%  vs v1: {diff_vs_v1:+.1f}%")
    if nobd_mean:
        diff_vs_v2 = (nobd_mean - v2_mean) / v2_mean * 100 if v2_mean else 0
        print(f"  NoBdry (无边界修正):         {nobd_mean:.0f} ± {nobd.get('std', 0):.0f}  n={nobd.get('n', 0)}  vs v2: {diff_vs_v2:+.1f}%  [边界修正效果: {-diff_vs_v2:+.1f}%]")
    if nogate_mean:
        diff_vs_v2 = (nogate_mean - v2_mean) / v2_mean * 100 if v2_mean else 0
        print(f"  NoGate (无EV门控):           {nogate_mean:.0f} ± {nogate.get('std', 0):.0f}  n={nogate.get('n', 0)}  vs v2: {diff_vs_v2:+.1f}%  [EV门控效果: {-diff_vs_v2:+.1f}%]")

print("\n" + "="*90)
print("完整数据覆盖表（缺失的标为❌）")
print("="*90)

header = f"{'算法':<35}"
for env in envs:
    header += f" {env.split('-')[0]:<12}"
print(header)
print("-"*90)

for algo in algos:
    row = f"{algo:<35}"
    for env in envs:
        d = all_data[env].get(algo, {})
        n = d.get("n", 0)
        mean = d.get("mean", None)
        if mean:
            row += f" {mean:.0f}({n})    "
        else:
            row += f" ❌({n})        "
    print(row)


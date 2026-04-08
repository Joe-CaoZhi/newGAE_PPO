#!/usr/bin/env python3
"""
HCGAE v4 最终统计分析
在V4框架内（update后eval）进行公平比较
"""
import json
import numpy as np
import os

from scipy import stats

envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
algos = ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v4']

v4_dir = 'results/V4FullExperiment'
icml_dir = 'results/ICMLExperiment'

print("=" * 80)
print("HCGAE v4 最终分析 — V4框架内公平比较（update后eval, 5seeds × 500K步）")
print("=" * 80)

all_results = {}
for env in envs:
    all_results[env] = {}
    for algo in algos:
        vals = []
        for s in range(5):
            path = f'{v4_dir}/{env}/{algo}/{algo}_s{s}.json'
            if os.path.exists(path):
                d = json.load(open(path))
                v = np.mean(d['eval_rewards'][-5:])
                vals.append(v)
        all_results[env][algo] = vals

print(f"\n{'环境':<18} {'方法':<25} {'均值':>8} {'SEM':>6} {'n':>3}")
print("-" * 65)
for env in envs:
    for algo in algos:
        vals = all_results[env].get(algo, [])
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/np.sqrt(len(vals)) if len(vals)>1 else 0
            print(f"{env:<18} {algo:<25} {m:>8.1f} {sem:>6.1f} {len(vals):>3}")
    print()

print("\n" + "=" * 80)
print("统计检验（v4 vs PPO 和 v4 vs v2）")
print("=" * 80)

print(f"\n{'环境':<15} {'比较':<22} {'Δ%':>8} {'p-值':>8} {'Cohen d':>9}")
print("-" * 66)

for env in envs:
    ppo_vals = all_results[env].get('Optimal_PPO', [])
    v2_vals  = all_results[env].get('Optimal_HCGAE_v2', [])
    v4_vals  = all_results[env].get('Optimal_HCGAE_v4', [])

    if ppo_vals and v4_vals:
        ppo_m = np.mean(ppo_vals)
        v4_m  = np.mean(v4_vals)
        pct = (v4_m - ppo_m) / abs(ppo_m) * 100
        u, p = stats.mannwhitneyu(v4_vals, ppo_vals, alternative='two-sided')
        n1, n2 = len(v4_vals), len(ppo_vals)
        d = (v4_m - ppo_m) / np.sqrt((np.var(v4_vals)*n1 + np.var(ppo_vals)*n2)/(n1+n2))
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{env:<15} {'v4 vs PPO':<22} {pct:>+8.1f}% {p:>8.3f}{star:<3} {d:>9.2f}")

    if v2_vals and v4_vals:
        v2_m = np.mean(v2_vals)
        v4_m = np.mean(v4_vals)
        pct = (v4_m - v2_m) / abs(v2_m) * 100
        u, p = stats.mannwhitneyu(v4_vals, v2_vals, alternative='two-sided')
        n1, n2 = len(v4_vals), len(v2_vals)
        d = (v4_m - v2_m) / np.sqrt((np.var(v4_vals)*n1 + np.var(v2_vals)*n2)/(n1+n2))
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{env:<15} {'v4 vs v2':<22} {pct:>+8.1f}% {p:>8.3f}{star:<3} {d:>9.2f}")
    print()

print("\n" + "=" * 80)
print("与ICMLExperiment的对比（eval timing参考）")
print("=" * 80)
print("\n（注：ICMLExperiment使用update前eval，V4框架使用update后eval，两者eval timing不同）")
print("（因此只比较各框架内部的相对改进，而不是跨框架的绝对数值）")
print()

for env in envs:
    v4_ppo = np.mean(all_results[env].get('Optimal_PPO', [0]))
    v4_v2  = np.mean(all_results[env].get('Optimal_HCGAE_v2', [0]))
    v4_v4  = np.mean(all_results[env].get('Optimal_HCGAE_v4', [0]))

    # ICML数据
    icml_ppo_vals = [np.mean(json.load(open(f'{icml_dir}/{env}/Optimal_PPO/Optimal_PPO_s{s}.json'))['eval_rewards'][-5:]) for s in range(5)]
    icml_v2_vals  = [np.mean(json.load(open(f'{icml_dir}/{env}/Optimal_HCGAE_v2/Optimal_HCGAE_v2_s{s}.json'))['eval_rewards'][-5:]) for s in range(5)]
    icml_ppo = np.mean(icml_ppo_vals)
    icml_v2  = np.mean(icml_v2_vals)

    pct_v2_icml = (icml_v2 - icml_ppo) / abs(icml_ppo) * 100
    pct_v4_v4  = (v4_v4 - v4_ppo) / abs(v4_ppo) * 100
    pct_v4_v2  = (v4_v4 - v4_v2) / abs(v4_v2) * 100 if v4_v2 > 0 else float('nan')
    pct_v2_v4f = (v4_v2 - v4_ppo) / abs(v4_ppo) * 100 if v4_ppo > 0 else float('nan')

    print(f"{env}:")
    print(f"  ICML: v2 vs PPO = {pct_v2_icml:+.1f}%  (v2={icml_v2:.0f}, PPO={icml_ppo:.0f})")
    print(f"  V4框架: v2 vs PPO = {pct_v2_v4f:+.1f}%  (v2={v4_v2:.0f}, PPO={v4_ppo:.0f})")
    print(f"  V4框架: v4 vs PPO = {pct_v4_v4:+.1f}%  (v4={v4_v4:.0f}, PPO={v4_ppo:.0f})")
    print(f"  V4框架: v4 vs v2  = {pct_v4_v2:+.1f}%  (v4={v4_v4:.0f}, v2={v4_v2:.0f})")
    print()

print("=" * 80)
print("论文表格格式（V4框架内，5 seeds × 500K steps, update后eval）")
print("=" * 80)
print()
print(f"| 方法 | Hopper-v4 | Walker2d-v4 | HalfCheetah-v4 |")
print(f"|---|:---:|:---:|:---:|")
for algo in algos:
    row = f"| **{algo}** |"
    for env in envs:
        vals = all_results[env].get(algo, [])
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/np.sqrt(len(vals)) if len(vals)>1 else 0
            row += f" {m:.0f} ± {sem:.0f} |"
        else:
            row += " — |"
    print(row)

# Delta rows
row_v4_ppo = "| Δ v4 vs PPO |"
row_v4_v2  = "| Δ v4 vs v2  |"
for env in envs:
    ppo_vals = all_results[env].get('Optimal_PPO', [])
    v2_vals  = all_results[env].get('Optimal_HCGAE_v2', [])
    v4_vals  = all_results[env].get('Optimal_HCGAE_v4', [])
    if ppo_vals and v4_vals:
        pct = (np.mean(v4_vals) - np.mean(ppo_vals)) / abs(np.mean(ppo_vals)) * 100
        row_v4_ppo += f" **{pct:+.1f}%** |"
    else:
        row_v4_ppo += " — |"
    if v2_vals and v4_vals:
        pct = (np.mean(v4_vals) - np.mean(v2_vals)) / abs(np.mean(v2_vals)) * 100
        row_v4_v2  += f" **{pct:+.1f}%** |"
    else:
        row_v4_v2  += " — |"
print(row_v4_ppo)
print(row_v4_v2)


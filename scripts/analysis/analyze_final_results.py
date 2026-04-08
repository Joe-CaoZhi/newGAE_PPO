#!/usr/bin/env python3
"""
Comprehensive analysis of all experiment results for paper update.
Analyzes: Optimal_PPO, Optimal_HCGAE, Optimal_HCGAE_v2, Standard_PPO
across Hopper-v4, Walker2d-v4, HalfCheetah-v4, and Ant-v4.
"""
import glob
import json

import numpy as np
from scipy import stats

BASE_DIR = "results/ICMLExperiment"

def load_results(env, algo):
    """Load all seed results for an env/algo pair."""
    files = sorted(glob.glob(f"{BASE_DIR}/{env}/{algo}/{algo}_s*.json"))
    results = []
    for f in files:
        try:
            d = json.load(open(f))
            er = d.get('eval_rewards', [])
            if er:
                final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                results.append(final)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    return results

def compute_stats(values):
    """Compute mean ± std."""
    if not values:
        return None, None
    return np.mean(values), np.std(values)

def mannwhitney_test(a, b):
    """Mann-Whitney U test."""
    if len(a) < 2 or len(b) < 2:
        return None, None
    stat, pval = stats.mannwhitneyu(a, b, alternative='two-sided')
    # Cohen's d
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0
    return pval, d

ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
ALGOS = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']

print("=" * 80)
print("COMPREHENSIVE RESULTS ANALYSIS")
print("=" * 80)

all_data = {}
for env in ENVS:
    all_data[env] = {}
    print(f"\n{'='*60}")
    print(f"Environment: {env}")
    print(f"{'='*60}")
    for algo in ALGOS:
        vals = load_results(env, algo)
        all_data[env][algo] = vals
        if vals:
            mean, std = compute_stats(vals)
            print(f"  {algo:<25} n={len(vals):2d}  {mean:7.1f} ± {std:6.1f}  seeds={[f'{x:.0f}' for x in vals]}")
        else:
            print(f"  {algo:<25} n= 0  NO DATA")

print("\n\n" + "=" * 80)
print("STATISTICAL COMPARISON: HCGAE vs Optimal_PPO (primary baseline)")
print("=" * 80)
for env in ENVS:
    opt_ppo = all_data[env].get('Optimal_PPO', [])
    print(f"\n{env}:")
    for algo in ['Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']:
        vals = all_data[env].get(algo, [])
        if not vals or not opt_ppo:
            continue
        pval, d = mannwhitney_test(vals, opt_ppo)
        improvement = (np.mean(vals) - np.mean(opt_ppo)) / np.mean(opt_ppo) * 100
        sig = "***" if pval and pval < 0.001 else ("**" if pval and pval < 0.01 else ("*" if pval and pval < 0.05 else "ns"))
        pval_str = f"{pval:.4f}" if pval is not None else "N/A"
        d_str = f"{d:+.2f}" if d is not None else "N/A"
        print(f"  {algo} vs Optimal_PPO: {improvement:+.1f}%, p={pval_str} {sig}, d={d_str}")

print("\n\n" + "=" * 80)
print("TABLE: MEAN ± STD SUMMARY FOR PAPER")
print("=" * 80)
algo_names = {
    'Standard_PPO': 'Standard PPO',
    'Optimal_PPO': 'Optimal PPO',
    'Optimal_HCGAE': 'HCGAE (v1)',
    'Optimal_HCGAE_SCR': 'HCGAE+SCR',
    'Optimal_HCGAE_v2': 'HCGAE (v2)',
}
header = f"{'Algorithm':<20} | {'Hopper-v4':>20} | {'Walker2d-v4':>20} | {'HalfCheetah-v4':>20} | {'Ant-v4':>20}"
print(header)
print("-" * len(header))
for algo in ALGOS:
    row = f"{algo_names.get(algo, algo):<20}"
    for env in ENVS:
        vals = all_data[env].get(algo, [])
        if vals:
            m, s = compute_stats(vals)
            row += f" | {m:8.0f} ± {s:6.0f}"
        else:
            row += f" | {'N/A':>20}"
    print(row)

print("\n\n" + "=" * 80)
print("HalfCheetah DETAILED ANALYSIS (v1 vs v2 comparison)")
print("=" * 80)
for algo in ['Optimal_HCGAE', 'Optimal_HCGAE_v2']:
    vals = all_data['HalfCheetah-v4'].get(algo, [])
    opt_ppo = all_data['HalfCheetah-v4'].get('Optimal_PPO', [])
    if vals and opt_ppo:
        diff = np.mean(vals) - np.mean(opt_ppo)
        pct = diff / np.mean(opt_ppo) * 100
        print(f"  {algo}: {np.mean(vals):.1f}  vs Optimal_PPO {np.mean(opt_ppo):.1f}  => {pct:+.1f}%")

print("\n\n" + "=" * 80)
print("KEY FINDINGS FOR PAPER")
print("=" * 80)
# v2 vs v1 improvement
v1_hc = all_data['HalfCheetah-v4'].get('Optimal_HCGAE', [])
v2_hc = all_data['HalfCheetah-v4'].get('Optimal_HCGAE_v2', [])
opt_hc = all_data['HalfCheetah-v4'].get('Optimal_PPO', [])
if v1_hc and v2_hc and opt_hc:
    v2_vs_v1 = (np.mean(v2_hc) - np.mean(v1_hc)) / np.mean(v1_hc) * 100
    v2_vs_opt = (np.mean(v2_hc) - np.mean(opt_hc)) / np.mean(opt_hc) * 100
    print(f"  HalfCheetah: v2 vs v1: {v2_vs_v1:+.1f}%")
    print(f"  HalfCheetah: v2 vs Optimal_PPO: {v2_vs_opt:+.1f}%")
    print(f"  HalfCheetah: v2 std={np.std(v2_hc):.1f} (n={len(v2_hc)}), v1 std={np.std(v1_hc):.1f} (n={len(v1_hc)})")


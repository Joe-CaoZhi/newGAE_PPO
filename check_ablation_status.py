#!/usr/bin/env python3
"""Check current status of all experiments and compute summaries."""
import json
import glob
import numpy as np
import os

def get_algo_stats(env, algo):
    files = glob.glob(f'results/ICMLExperiment/{env}/{algo}/*_metrics.json')
    if not files:
        return None, 0
    means = []
    for f in files:
        try:
            d = json.load(open(f))
            v = d.get('mean_reward', d.get('final_mean', None))
            if v is not None:
                means.append(v)
        except:
            pass
    if not means:
        return None, 0
    return means, len(means)

print("=" * 70)
print("EXPERIMENT STATUS CHECK")
print("=" * 70)

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2',
         'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

for env in envs:
    print(f"\n=== {env} ===")
    for algo in algos:
        means, n = get_algo_stats(env, algo)
        if n == 0:
            continue
        m = np.mean(means)
        s = np.std(means)
        print(f"  {algo:<30s} ({n}/5): {m:.0f} ± {s:.0f}")

print("\n" + "=" * 70)
print("ABLATION ANALYSIS")
print("=" * 70)
print("\nKey question: Which v2 fixes are effective?")
print("  v1 = baseline (c_mc>=0.1, no boundary corr, no EV gate)")
print("  v2 = all fixes (c_mc>=0.1 + boundary corr + EV gate)")
print("  v2_NoBdry = no boundary correction (EV gate only)")
print("  v2_NoGate = no EV gate (boundary correction only)")

print("\nv2 vs v1 comparison:")
for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    v1, n1 = get_algo_stats(env, 'Optimal_HCGAE')
    v2, n2 = get_algo_stats(env, 'Optimal_HCGAE_v2')
    if v1 and v2:
        pct = (np.mean(v2) - np.mean(v1)) / abs(np.mean(v1)) * 100
        print(f"  {env}: v1={np.mean(v1):.0f}(n={n1}) -> v2={np.mean(v2):.0f}(n={n2}) [{pct:+.1f}%]")
    elif v1:
        print(f"  {env}: v1={np.mean(v1):.0f}(n={n1}) -> v2=MISSING")

print("\nFile counts per env/algo:")
for env in envs:
    for algo in ['Optimal_HCGAE', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']:
        files = glob.glob(f'results/ICMLExperiment/{env}/{algo}/*_metrics.json')
        if files:
            print(f"  {env}/{algo}: {len(files)} files")


#!/usr/bin/env python3
"""Check different std calculation methods to match paper values."""
import json
import os

import numpy as np

# Focus on Hopper Optimal PPO (paper says mean=1598, std=133, sem=60)
path = 'results/ICMLExperiment/Hopper-v4/Optimal_PPO/'
per_seed_finals = []
per_seed_means = []

print("=== Hopper-v4 Optimal PPO seed details ===")
for fname in sorted(os.listdir(path)):
    if fname.endswith('.json'):
        data = json.load(open(os.path.join(path, fname)))
        for k in ['eval_rewards', 'eval_returns']:
            if k in data and data[k]:
                evals = data[k]
                last5_mean = np.mean(evals[-5:])
                last1 = evals[-1]
                print(f"  {fname}: last5_mean={last5_mean:.1f}, last1={last1:.1f}, n_evals={len(evals)}")
                per_seed_finals.append(last5_mean)
                per_seed_means.append(last1)
                break

print(f"\nPer-seed finals (mean of last 5): {[f'{v:.1f}' for v in per_seed_finals]}")
print(f"Overall mean: {np.mean(per_seed_finals):.1f}")
print(f"Std (ddof=1): {np.std(per_seed_finals, ddof=1):.1f}")
print(f"Std (ddof=0): {np.std(per_seed_finals, ddof=0):.1f}")
print(f"SEM: {np.std(per_seed_finals, ddof=1)/np.sqrt(len(per_seed_finals)):.1f}")

# Check if paper uses ddof=0
n = len(per_seed_finals)
print(f"\nIf paper uses std (ddof=0): {np.std(per_seed_finals, ddof=0):.1f}")
print(f"If paper uses std / sqrt(n-1): {np.std(per_seed_finals, ddof=1)/np.sqrt(n-1):.1f}")

# Check Walker2d HCGAE
print("\n=== Walker2d-v4 HCGAE ===")
path2 = 'results/ICMLExperiment/Walker2d-v4/Optimal_HCGAE_v2/'
if os.path.exists(path2):
    vals = []
    for fname in sorted(os.listdir(path2)):
        if fname.endswith('.json'):
            data = json.load(open(os.path.join(path2, fname)))
            for k in ['eval_rewards', 'eval_returns']:
                if k in data and data[k]:
                    vals.append(np.mean(data[k][-5:]))
                    break
    print(f"Values: {[f'{v:.1f}' for v in vals]}")
    print(f"Std (ddof=1): {np.std(vals, ddof=1):.1f}")
    print(f"Std (ddof=0): {np.std(vals, ddof=0):.1f}")
    print(f"SEM: {np.std(vals, ddof=1)/np.sqrt(len(vals)):.1f}")
    print(f"Paper (en, SEM): 314; Paper (zh, std): 702")


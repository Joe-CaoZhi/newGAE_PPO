#!/usr/bin/env python3
"""检查AlignedExperiment的配置和训练详情"""
import json
import numpy as np
from pathlib import Path

for env in ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']:
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        p = Path(f'results/AlignedExperiment/{env}/{algo}/{algo}_s0.json')
        if p.exists():
            d = json.load(open(p))
            print(f"\n{env}/{algo}:")
            print(f"  Keys: {[k for k in d.keys()]}")
            print(f"  total_steps: {d.get('total_steps', 'N/A')}")
            print(f"  num_evals: {len(d.get('eval_rewards', []))}")
            evals = d.get('eval_rewards', [])
            if evals:
                print(f"  first 5 evals: {[round(x) for x in evals[:5]]}")
                print(f"  last 5 evals: {[round(x) for x in evals[-5:]]}")
            # check config
            for cfg_key in ['config', 'hyperparams', 'params', 'algorithm']:
                if cfg_key in d:
                    print(f"  {cfg_key}: {d[cfg_key]}")
            break  # just first seed
    break  # just HalfCheetah

# Compare ICMLExperiment
for env in ['HalfCheetah-v4']:
    for algo in ['Optimal_PPO']:
        p = Path(f'results/ICMLExperiment/{env}/{algo}/{algo}_s0.json')
        if p.exists():
            d = json.load(open(p))
            print(f"\n=== ICMLExperiment {env}/{algo}:")
            print(f"  Keys: {[k for k in d.keys()]}")
            print(f"  total_steps: {d.get('total_steps', 'N/A')}")
            print(f"  num_evals: {len(d.get('eval_rewards', []))}")
            evals = d.get('eval_rewards', [])
            if evals:
                print(f"  first 5 evals: {[round(x) for x in evals[:5]]}")
                print(f"  last 5 evals: {[round(x) for x in evals[-5:]]}")

# Also print detailed HC comparison
print("\n\n=== HalfCheetah Aligned Detail ===")
for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']:
    scores = []
    for s in range(5):
        p = Path(f'results/AlignedExperiment/HalfCheetah-v4/{algo}/{algo}_s{s}.json')
        if p.exists():
            d = json.load(open(p))
            evals = d.get('eval_rewards', [])
            if evals:
                scores.append(evals[-5:])
    if scores:
        all_final = [np.mean(s) for s in scores]
        print(f"  {algo:40s}: {np.mean(all_final):.1f} ± {np.std(all_final)/np.sqrt(len(all_final)):.1f}")
        # Also per-seed last 5 mean
        per_seed = [round(np.mean(s)) for s in scores]
        print(f"    per-seed: {per_seed}")

print("\n\n=== HalfCheetah ICML Detail ===")
for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']:
    scores = []
    for s in range(5):
        p = Path(f'results/ICMLExperiment/HalfCheetah-v4/{algo}/{algo}_s{s}.json')
        if p.exists():
            d = json.load(open(p))
            evals = d.get('eval_rewards', [])
            if evals:
                scores.append(evals[-5:])
    if scores:
        all_final = [np.mean(s) for s in scores]
        print(f"  {algo:40s}: {np.mean(all_final):.1f} ± {np.std(all_final)/np.sqrt(len(all_final)):.1f}")
        per_seed = [round(np.mean(s)) for s in scores]
        print(f"    per-seed: {per_seed}")


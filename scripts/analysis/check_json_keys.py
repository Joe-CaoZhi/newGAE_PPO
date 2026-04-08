#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
algos = ['Optimal_PPO', 'Optimal_HCGAE_v2']

# Check keys in both experiments
print("=== AlignedExperiment JSON structure ===")
p = Path('results/AlignedExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json')
d = json.load(open(p))
print("Keys:", list(d.keys()))
evals = d.get('eval_rewards', [])
print(f"eval_rewards: n={len(evals)}, range=[{min(evals):.1f}, {max(evals):.1f}]")
print(f"First 3: {evals[:3]}")
print(f"Last 3: {evals[-3:]}")

print()
print("=== ICMLExperiment JSON structure ===")
p2 = Path('results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json')
if p2.exists():
    d2 = json.load(open(p2))
    print("Keys:", list(d2.keys()))
    evals2 = d2.get('eval_rewards', [])
    print(f"eval_rewards: n={len(evals2)}, range=[{min(evals2):.1f}, {max(evals2):.1f}]")
    print(f"First 3: {evals2[:3]}")
    print(f"Last 3: {evals2[-3:]}")

# Compare seed-by-seed for HalfCheetah
print()
print("=== HalfCheetah-v4 Seed-by-Seed Detail ===")
for algo in algos:
    print(f"\n{algo}:")
    for exp_dir in ['AlignedExperiment', 'ICMLExperiment']:
        vals = []
        for s in range(5):
            p = Path(f'results/{exp_dir}/HalfCheetah-v4/{algo}/{algo}_s{s}.json')
            if p.exists():
                d = json.load(open(p))
                evals = d.get('eval_rewards', [])
                final5 = float(np.mean(evals[-5:])) if evals else None
                vals.append(final5)
        if vals:
            print(f"  {exp_dir}: mean={np.mean(vals):.1f}, seeds={[round(v,1) if v else None for v in vals]}")

# Compute model complexity metrics
print()
print("=== Model Complexity Comparison ===")
for exp_dir in ['AlignedExperiment', 'ICMLExperiment']:
    p = Path(f'results/{exp_dir}/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s0.json')
    if p.exists():
        d = json.load(open(p))
        print(f"\n{exp_dir} - Optimal_PPO_s0 config:")
        for key in ['hidden_size', 'network_arch', 'hidden_dim', 'n_steps', 'total_steps', 'use_obs_norm', 'use_adv_norm']:
            if key in d:
                print(f"  {key}: {d[key]}")
        # Look for config sub-dict
        if 'config' in d:
            print(f"  config: {d['config']}")
        if 'hyperparams' in d:
            print(f"  hyperparams: {d['hyperparams']}")
        if 'args' in d:
            print(f"  args: {d['args']}")

# Compute relative HCGAE gain in both experiments
print()
print("=== HCGAE v2 vs Optimal PPO Relative Gain ===")
for env in envs:
    print(f"\n{env}:")
    for exp_dir in ['AlignedExperiment', 'ICMLExperiment']:
        ppo_vals, hcgae_vals = [], []
        for s in range(5):
            for algo, store in [('Optimal_PPO', ppo_vals), ('Optimal_HCGAE_v2', hcgae_vals)]:
                p = Path(f'results/{exp_dir}/{env}/{algo}/{algo}_s{s}.json')
                if p.exists():
                    d = json.load(open(p))
                    evals = d.get('eval_rewards', [])
                    if evals:
                        store.append(np.mean(evals[-5:]))
        if ppo_vals and hcgae_vals:
            gain = (np.mean(hcgae_vals) - np.mean(ppo_vals)) / np.mean(ppo_vals) * 100
            print(f"  {exp_dir}: PPO={np.mean(ppo_vals):.1f}, HCGAE={np.mean(hcgae_vals):.1f}, Δ={gain:+.1f}%")


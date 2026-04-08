#!/usr/bin/env python3
"""检查V4实验的学习曲线真实性"""
import json

import numpy as np

print("=== HalfCheetah-v4 V4实验学习曲线 ===")
for s in range(5):
    path = f'results/V4FullExperiment/HalfCheetah-v4/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s{s}.json'
    with open(path) as f:
        d = json.load(f)
    evals = d['eval_rewards']
    steps = d['eval_steps']
    n = len(evals)
    print(f"s{s}: n={n}, first={evals[0]:.1f}, @50%={evals[n//2]:.1f}, last5={np.mean(evals[-5:]):.1f}, max={max(evals):.1f}, total_steps={d.get('total_steps','?')}")
    print(f"     steps: {steps[0]} -> {steps[-1]}, last_step={steps[-1]}")

print("\n=== ICMLExperiment HalfCheetah Optimal_PPO ===")
for s in range(5):
    path = f'results/ICMLExperiment/HalfCheetah-v4/Optimal_PPO/Optimal_PPO_s{s}.json'
    with open(path) as f:
        d = json.load(f)
    evals = d['eval_rewards']
    steps = d['eval_steps']
    n = len(evals)
    print(f"s{s}: n={n}, first={evals[0]:.1f}, @50%={evals[n//2]:.1f}, last5={np.mean(evals[-5:]):.1f}, max={max(evals):.1f}, total_steps={d.get('total_steps','?')}")

print("\n=== ICMLExperiment Hopper Optimal_PPO ===")
for s in range(5):
    path = f'results/ICMLExperiment/Hopper-v4/Optimal_PPO/Optimal_PPO_s{s}.json'
    with open(path) as f:
        d = json.load(f)
    evals = d['eval_rewards']
    n = len(evals)
    print(f"s{s}: n={n}, last5={np.mean(evals[-5:]):.1f}, max={max(evals):.1f}")

print("\n=== V4FullExperiment Hopper (完成的seeds) ===")
for s in range(5):
    import os
    path = f'results/V4FullExperiment/Hopper-v4/Optimal_HCGAE_v4/Optimal_HCGAE_v4_s{s}.json'
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        evals = d['eval_rewards']
        n = len(evals)
        print(f"s{s}: n={n}, last5={np.mean(evals[-5:]):.1f}, max={max(evals):.1f}, total_steps={d.get('total_steps','?')}")


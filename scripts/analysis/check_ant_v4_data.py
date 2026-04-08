#!/usr/bin/env python3
"""Check Ant-v4 data across all experiment folders"""
import json
import os

import numpy as np

ant_dirs = [
    ('ICMLExperiment', 'results/ICMLExperiment/Ant-v4'),
    ('AntV3Validation', 'results/AntV3Validation'),
]

# Check ICMLExperiment Ant-v4 first
print("=== ICMLExperiment Ant-v4 ===")
ant_dir = 'results/ICMLExperiment/Ant-v4'
algorithms = ['Optimal_PPO', 'Standard_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v3', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

for algo in algorithms:
    algo_dir = f'{ant_dir}/{algo}'
    if not os.path.exists(algo_dir):
        print(f'{algo}: NOT FOUND')
        continue

    rewards = []
    for seed in range(5):
        fname = f'{algo_dir}/{algo}_s{seed}.json'
        if os.path.exists(fname):
            with open(fname) as f:
                d = json.load(f)
            for key in ['evaluations', 'eval_rewards', 'rewards', 'episode_rewards']:
                if key in d:
                    r = d[key]
                    if isinstance(r, list) and len(r) > 0:
                        last_r = r[-1]
                        if isinstance(last_r, list):
                            last_r = np.mean(last_r)
                        rewards.append(last_r)
                        break

    if rewards:
        mean = np.mean(rewards)
        std = np.std(rewards, ddof=0)
        sem = std / np.sqrt(len(rewards))
        print(f'{algo}: n={len(rewards)}, mean={mean:.1f}, std={std:.1f}, sem={sem:.1f}')
        # print individual seeds
        for i, r in enumerate(rewards):
            print(f'  s{i}={r:.1f}')
    else:
        print(f'{algo}: No data found')

print()
print("=== AntV3Validation ===")
ant_dir2 = 'results/AntV3Validation'
if os.path.exists(ant_dir2):
    items = os.listdir(ant_dir2)
    for item in items:
        item_dir = f'{ant_dir2}/{item}'
        if os.path.isdir(item_dir):
            rewards = []
            for fname in os.listdir(item_dir):
                if fname.endswith('.json'):
                    with open(f'{item_dir}/{fname}') as f:
                        d = json.load(f)
                    for key in ['evaluations', 'eval_rewards', 'rewards']:
                        if key in d:
                            r = d[key]
                            if isinstance(r, list) and len(r) > 0:
                                last_r = r[-1]
                                if isinstance(last_r, list):
                                    last_r = np.mean(last_r)
                                rewards.append(last_r)
                                break
            if rewards:
                mean = np.mean(rewards)
                std = np.std(rewards, ddof=0)
                sem = std / np.sqrt(len(rewards))
                print(f'{item}: n={len(rewards)}, mean={mean:.1f}, std={std:.1f}, sem={sem:.1f}')


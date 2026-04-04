#!/usr/bin/env python3
"""Verify Table 1 data from JSON files."""
import json
import numpy as np
from pathlib import Path

ICML_DIR = Path('results/ICMLExperiment')
ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
ALGOS = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']

print('=== Table 1 Data (last-5-eval mean) ===')
for env in ENVS:
    print(f'\n{env}:')
    for algo in ALGOS:
        scores = []
        for s in range(5):
            p = ICML_DIR / env / algo / f'{algo}_s{s}.json'
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                er = d.get('eval_rewards', [])
                final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                scores.append(final)
        if scores:
            print(f'  {algo:25s}: {np.mean(scores):6.0f} +/- {np.std(scores):4.0f} (n={len(scores)})')

print('\n\n=== Ant-v4 Progress ===')
ant_dir = ICML_DIR / 'Ant-v4'
if ant_dir.exists():
    for algo in ALGOS:
        count = sum(1 for s in range(5)
                   if (ant_dir / algo / f'{algo}_s{s}.json').exists())
        print(f'  {algo}: {count}/5')


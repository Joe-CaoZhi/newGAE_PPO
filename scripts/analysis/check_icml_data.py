#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE_v2']
base = Path('/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment')

for env in envs:
    print(f'\n=== {env} ===')
    for algo in algos:
        means = []
        for s in range(5):
            fp = base / env / algo / f'{algo}_s{s}.json'
            if fp.exists():
                d = json.load(open(fp))
                er = d.get('eval_rewards', [])
                steps = d.get('total_steps', 0)
                if er and steps > 100000:
                    val = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                    means.append(val)
        if means:
            print(f'  {algo}: mean={np.mean(means):.1f} +/- {np.std(means):.1f} (n={len(means)})')
        else:
            print(f'  {algo}: no complete data')


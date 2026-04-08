#!/usr/bin/env python3
"""显示当前实验的详细中间结果"""
import json
from pathlib import Path

import numpy as np

BASE = Path('results/ICMLExperiment')
ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
ALGOS = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']

print('=== 详细数据（各seed最终奖励）===')
for env in ENVS:
    print(f'\n{env}:')
    for algo in ALGOS:
        algo_dir = BASE / env / algo
        if not algo_dir.exists():
            continue
        vals = []
        for fp in sorted(algo_dir.glob('*.json')):
            d = json.load(open(fp))
            er = d.get('eval_rewards', [])
            val = float(np.mean(er[-5:])) if er else d.get('final_reward', 0)
            vals.append(val)
        if vals:
            n = len(vals)
            std = np.std(vals, ddof=1) if n > 1 else 0
            print(f'  {algo:<22}: n={n}  [{", ".join(str(round(v)) for v in vals)}]  mean={np.mean(vals):.0f}±{std:.0f}')
        else:
            print(f'  {algo:<22}: (no data yet)')


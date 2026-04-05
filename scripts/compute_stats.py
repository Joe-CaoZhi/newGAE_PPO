#!/usr/bin/env python3
"""计算 ICMLExperiment 所有算法的最终性能统计"""
import glob
import json

import numpy as np

ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
ALGOS = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR',
         'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

for env in ENVS:
    print(f'\n=== {env} ===')
    for algo in ALGOS:
        files = sorted(glob.glob(f'results/ICMLExperiment/{env}/{algo}/{algo}_s*.json'))
        if not files:
            continue
        finals = []
        for f in files:
            d = json.load(open(f))
            rets = d.get('eval_rewards', [])
            if rets:
                finals.append(np.mean(rets[-5:]))
        if finals:
            print(f'  {algo}: mean={np.mean(finals):.1f} std={np.std(finals):.1f} n={len(finals)}')


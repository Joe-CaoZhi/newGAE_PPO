#!/usr/bin/env python3
"""Verify Table 1 data against JSON files."""
import json
import os

import numpy as np

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
methods_map = {
    'Standard_PPO': 'Std PPO',
    'Optimal_PPO': 'Opt. PPO',
    'Optimal_HCGAE': 'HCGAE-Base',
    'Optimal_HCGAE_v2': 'HCGAE',
    'Optimal_HCGAE_SCR': 'HCGAE+SCR',
}

for env in envs:
    print(f'\n=== {env} ===')
    for method_dir, method_name in methods_map.items():
        path = f'results/ICMLExperiment/{env}/{method_dir}/'
        if not os.path.exists(path):
            continue
        rewards = []
        for fname in sorted(os.listdir(path)):
            if fname.endswith('.json'):
                data = json.load(open(os.path.join(path, fname)))
                for k in ['eval_rewards', 'eval_returns']:
                    if k in data and data[k]:
                        last5 = data[k][-5:]
                        rewards.append(np.mean(last5))
                        break
        if rewards:
            m = np.mean(rewards)
            s = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
            sem = s / np.sqrt(len(rewards))
            print(f'  {method_name} (n={len(rewards)}): mean={m:.0f}, std={s:.0f} [zh_paper], sem={sem:.0f} [en_paper]')


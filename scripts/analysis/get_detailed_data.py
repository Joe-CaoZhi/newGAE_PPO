#!/usr/bin/env python3
"""Get detailed per-seed data for specific experiments"""
import json
import numpy as np
import os

base = 'results/ICMLExperiment'

# Experiments to check in detail
targets = [
    ('Ant-v4', 'Optimal_HCGAE_v2'),
    ('Ant-v4', 'Optimal_HCGAE_v2_NoBdry'),
    ('Ant-v4', 'Optimal_HCGAE_SCR'),
    ('Walker2d-v4', 'Optimal_HCGAE_v2'),
    ('HalfCheetah-v4', 'Optimal_HCGAE_v2_NoBdry'),
]

for env, algo in targets:
    d = f'{base}/{env}/{algo}'
    if not os.path.exists(d):
        print(f'{env}/{algo}: MISSING')
        continue
    files = sorted([f for f in os.listdir(d) if f.endswith('.json')])
    print(f'\n{env}/{algo}: ({len(files)} files)')
    per_seed = []
    for f in files:
        try:
            data = json.load(open(f'{d}/{f}'))
            evr = data.get('eval_rewards', [])
            if evr:
                val = float(np.mean(evr[-5:]))
                seed = f.split('_s')[-1].replace('.json', '')
                print(f'  s{seed}: {val:.1f}')
                per_seed.append(val)
        except Exception as e:
            print(f'  ERROR {f}: {e}')
    if per_seed:
        print(f'  --> mean={np.mean(per_seed):.1f} +/- {np.std(per_seed):.1f} (n={len(per_seed)})')


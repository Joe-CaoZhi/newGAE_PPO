#!/usr/bin/env python3
"""检查旧格式的 DCPPO 结果文件"""
import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path('/Users/joe-caozhi/newGAE_ppo/results/MultiEnv_DCPPO')

for env in ['Hopper-v4', 'Walker2d-v4']:
    print(f'\n[{env}]')
    for variant in ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']:
        p = RESULTS_DIR / env / variant / f'{variant}_metrics.json'
        if p.exists():
            d = json.load(open(p))
            evr = d.get('eval_rewards', [])
            n = len(evr)
            mean5 = np.mean(evr[-5:]) if len(evr)>=5 else 0
            print(f'  {variant}: {n} evals, last5_mean={mean5:.1f}')
        else:
            print(f'  {variant}: NOT FOUND')


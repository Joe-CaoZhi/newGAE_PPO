#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

base = Path('results/V4FullExperiment')
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    p = base / env / 'Optimal_HCGAE_v4' / 'Optimal_HCGAE_v4_s0.json'
    if p.exists():
        d = json.load(open(p))
        er = d.get('eval_rewards', [])
        if er:
            final = np.mean(er[-5:])
            elapsed = d.get('elapsed_s', '?')
            print(f'{env}: s0 final5={final:.1f}  elapsed={elapsed}s  n_evals={len(er)}')
    else:
        print(f'{env}: no data yet')


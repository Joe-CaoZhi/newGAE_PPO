#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

BASE = Path('results/ICMLExperiment')
d = BASE / 'Walker2d-v4' / 'Optimal_HCGAE_v2_NoBdry'
if d.exists():
    files = sorted(d.glob('*.json'))
    scores = []
    for f in files:
        try:
            data = json.load(open(f))
            if 'eval_rewards' in data and len(data['eval_rewards']) > 0:
                scores.append(round(float(np.mean(data['eval_rewards'][-5:])), 0))
        except:
            pass
    print(f'Walker NoBdry: n={len(scores)}, seeds={scores}')
    if scores:
        print(f'  mean={np.mean(scores):.0f}+/-{np.std(scores):.0f}')
else:
    print('dir not found')


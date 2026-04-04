import json, os, numpy as np
from pathlib import Path
base = Path('results/MultiSeedPower/Walker2d-v4/HCGAE_Imp12_SCR')
for i in range(1, 11):
    fn = base / f'HCGAE_Imp12_s{i}_metrics.json'
    if fn.exists():
        with open(fn) as f:
            d = json.load(f)
        er = d.get('eval_rewards', [])
        final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er)) if er else float('nan')
        print(f's{i}: final={final:.1f}, n_eval={len(er)}')
    else:
        print(f's{i}: NOT FOUND')


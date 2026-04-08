#!/usr/bin/env python3
"""Compare old Standard_PPO results vs new ICML Standard_PPO to check consistency."""
import glob
import json
import numpy as np

icml_base = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment'
print("="*60)
print("New ICML Standard_PPO (500k steps, seeds 0-4)")
print("="*60)
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    files = sorted(glob.glob(f'{icml_base}/{env}/Standard_PPO/*.json'))
    vals = []
    for f in files:
        d = json.load(open(f))
        er = d.get('eval_rewards', [])
        v = float(np.mean(er[-5:])) if len(er)>=5 else float(np.mean(er))
        vals.append(round(v, 1))
    if vals:
        print(f"  {env:<16}: n={len(vals)}  mean={np.mean(vals):.0f}  std={np.std(vals):.0f}  vals={vals}")

print()
print("="*60)
print("Old MultiSeedPower Standard_PPO (1M steps, seeds 1-10)")
print("="*60)
old = json.load(open('/Users/joe-caozhi/newGAE_ppo/results/MultiSeedPower/multiseed_summary_n10.json'))
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    d = old[env]['Standard_PPO']
    print(f"  {env:<16}: n={d['n_seeds']}  mean={d['mean']:.0f}  std={d['std']:.0f}")
    print(f"                   seeds={[round(x,0) for x in d['seeds']]}")


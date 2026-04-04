#!/usr/bin/env python3
"""Check learning curves for Hopper to diagnose Optimal_PPO underperformance."""
import glob
import json
import numpy as np

base = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment'
for env in ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']:
    print(f"\n{env}:")
    for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']:
        files = sorted(glob.glob(f'{base}/{env}/{algo}/*.json'))
        if not files:
            continue
        for f in files:
            d = json.load(open(f))
            er = d.get('eval_rewards', [])
            if not er:
                continue
            n = len(er)
            early = np.mean(er[:min(5, n)])
            mid = np.mean(er[n//3:2*n//3]) if n >= 3 else early
            late = np.mean(er[-5:]) if n >= 5 else np.mean(er)
            seed = d.get('seed', '?')
            elapsed = d.get('elapsed_s', 0)
            print(f"  {algo:<22} s{seed}: early={early:.0f} mid={mid:.0f} late={late:.0f} ({elapsed:.0f}s)")


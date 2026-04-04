#!/usr/bin/env python3
import json, glob, numpy as np

envs = ['Hopper-v4','Walker2d-v4','HalfCheetah-v4','Ant-v4']
algos = ['Standard_PPO','Optimal_PPO','Optimal_HCGAE','Optimal_HCGAE_v2','Optimal_HCGAE_SCR']

for env in envs:
    print(f'=== {env} ===')
    for algo in algos:
        files = glob.glob(f'results/ICMLExperiment/{env}/{algo}/*.json')
        if not files:
            continue
        vals = []
        for f in files:
            d = json.load(open(f))
            # Try different key names
            v = d.get('mean_reward', d.get('final_mean', None))
            if v is None:
                er = d.get('eval_returns', None)
                if er and len(er) > 0:
                    v = float(np.mean(er[-5:]))
            if v is not None:
                vals.append(v)
        if vals:
            print(f'  {algo}({len(vals)}s): {np.mean(vals):.0f} +/- {np.std(vals):.0f}')
        else:
            print(f'  {algo}({len(files)} files, no readable vals)')
            # Show one file's keys
            d = json.load(open(files[0]))
            print(f'    keys: {list(d.keys())[:8]}')


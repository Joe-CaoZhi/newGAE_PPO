import json
import numpy as np
import os

envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2']

print("ICMLExperiment 完整数据 (final5 均值 ± SEM)")
for env in envs:
    print(f"\n{env}:")
    ppo_m = None
    for algo in algos:
        vals = []
        for s in range(5):
            path = f'results/ICMLExperiment/{env}/{algo}/{algo}_s{s}.json'
            if os.path.exists(path):
                d = json.load(open(path))
                v = np.mean(d['eval_rewards'][-5:])
                vals.append(v)
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/np.sqrt(len(vals))
            if algo == 'Optimal_PPO':
                ppo_m = m
            delta = ""
            if ppo_m and algo != 'Optimal_PPO' and algo != 'Standard_PPO':
                pct = (m - ppo_m) / abs(ppo_m) * 100
                delta = f" (Δ vs PPO: {pct:+.1f}%)"
            print(f"  {algo:<30} = {m:.1f} ± {sem:.1f} (n={len(vals)}){delta}")


import json
import numpy as np
import os

envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4']
algos = ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v4']

print("=== V4FullExperiment 当前进度 ===")
for env in envs:
    print(f"\n{env}:")
    for algo in algos:
        vals = []
        for s in range(5):
            path = f'results/V4FullExperiment/{env}/{algo}/{algo}_s{s}.json'
            if os.path.exists(path):
                d = json.load(open(path))
                v = np.mean(d['eval_rewards'][-5:])
                vals.append(v)
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/np.sqrt(len(vals)) if len(vals) > 1 else 0
            first_eval = json.load(open(f'results/V4FullExperiment/{env}/{algo}/{algo}_s0.json'))['eval_rewards'][0]
            print(f"  {algo:<30} n={len(vals)}: mean={m:.1f} ± {sem:.1f}  [first_eval s0={first_eval:.1f}]")
        else:
            print(f"  {algo:<30} 无数据")

print("\n=== ICMLExperiment 参考基准 ===")
for env in envs:
    print(f"\n{env}:")
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        vals = []
        for s in range(5):
            path = f'results/ICMLExperiment/{env}/{algo}/{algo}_s{s}.json'
            if os.path.exists(path):
                d = json.load(open(path))
                v = np.mean(d['eval_rewards'][-5:])
                vals.append(v)
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/np.sqrt(len(vals)) if len(vals) > 1 else 0
            first_eval = json.load(open(f'results/ICMLExperiment/{env}/{algo}/{algo}_s0.json'))['eval_rewards'][0]
            print(f"  {algo:<30} n={len(vals)}: mean={m:.1f} ± {sem:.1f}  [first_eval s0={first_eval:.1f}]")


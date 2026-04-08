import json
import os

import numpy as np


def load_seeds(base_path):
    if not os.path.exists(base_path):
        return []
    files = sorted([f for f in os.listdir(base_path) if f.endswith('.json')])
    scores = []
    for f in files:
        try:
            with open(os.path.join(base_path, f)) as fp:
                d = json.load(fp)
            for key in ['eval_rewards', 'mean_reward', 'final_score', 'episode_rewards']:
                if key in d:
                    v = d[key]
                    if isinstance(v, list) and len(v) >= 5:
                        scores.append(np.mean(v[-5:]))
                    elif isinstance(v, (int, float)):
                        scores.append(float(v))
                    break
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    return scores

# ICMLExperiment all environments
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
algos = ['Optimal_PPO', 'Standard_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_SCR',
         'Optimal_HCGAE_v3', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

print('=== ICMLExperiment Results ===')
icml_results = {}
for env in envs:
    print(f'\n{env}:')
    icml_results[env] = {}
    for algo in algos:
        path = f'results/ICMLExperiment/{env}/{algo}'
        scores = load_seeds(path)
        if scores:
            m = np.mean(scores)
            s = np.std(scores, ddof=0)
            icml_results[env][algo] = {'mean': m, 'std': s, 'n': len(scores)}
            print(f'  {algo}: {m:.1f} ± {s:.1f} (n={len(scores)})')

print('\n\n=== V4FullExperiment Results (post-update eval) ===')
v4_results = {}
algos_v4 = ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v4']
envs_v4 = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
for env in envs_v4:
    print(f'\n{env}:')
    v4_results[env] = {}
    for algo in algos_v4:
        path = f'results/V4FullExperiment/{env}/{algo}'
        scores = load_seeds(path)
        if scores:
            m = np.mean(scores)
            s = np.std(scores, ddof=0)
            v4_results[env][algo] = {'mean': m, 'std': s, 'n': len(scores)}
            print(f'  {algo}: {m:.1f} ± {s:.1f} (n={len(scores)})')

print('\n\n=== AntV3Validation Results ===')
algos_ant = ['Optimal_PPO', 'Standard_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v3',
             'Optimal_HCGAE_v3_NoClamp', 'Optimal_HCGAE_v3_NoVWGate', 'Optimal_HCGAE_v3_NoBdryPrior']
for algo in algos_ant:
    path = f'results/AntV3Validation/{algo}'
    scores = load_seeds(path)
    if scores:
        m = np.mean(scores)
        s = np.std(scores, ddof=0)
        print(f'  {algo}: {m:.1f} ± {s:.1f} (n={len(scores)})')

print('\n\n=== Comparing HCGAE variants on Ant-v4 ===')
print('ICMLExperiment Ant-v4 (main results):')
for algo, data in icml_results.get('Ant-v4', {}).items():
    ppo_mean = icml_results['Ant-v4'].get('Optimal_PPO', {}).get('mean', 1)
    pct = (data['mean'] - ppo_mean) / ppo_mean * 100
    print(f'  {algo}: {data["mean"]:.1f} ± {data["std"]:.1f} ({pct:+.1f}% vs OptPPO)')

print('\n\n=== EVConvergenceStudy check ===')
ev_path = 'results/EVConvergenceStudy'
if os.path.exists(ev_path):
    files = [f for f in os.listdir(ev_path) if f.endswith('.json')]
    print(f'Found {len(files)} JSON files in EVConvergenceStudy')
    if files:
        for f in files[:3]:
            print(f'  {f}')

print('\n\n=== Check if MultiSeedPower has EV data ===')
ms_path = 'results/MultiSeedPower'
if os.path.exists(ms_path):
    files = []
    for root, dirs, fs in os.walk(ms_path):
        for fn in fs:
            if fn.endswith('.json'):
                files.append(os.path.join(root, fn))
    for f in files:
        print(f'  {f}')


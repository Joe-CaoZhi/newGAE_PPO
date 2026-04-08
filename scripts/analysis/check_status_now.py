import json
import numpy as np
import os


def get_mean_reward(fpath):
    with open(fpath) as f:
        d = json.load(f)
    if 'eval_rewards' in d:
        arr = d['eval_rewards']
        return np.mean(arr[-5:]) if len(arr) >= 5 else np.mean(arr)
    for k in ['final_reward', 'mean_reward', 'final_mean']:
        if k in d:
            return d[k]
    return None

base = 'results/ICMLExperiment'
envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4', 'Ant-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

for env in envs:
    print(f'\n=== {env} ===')
    for algo in algos:
        d = os.path.join(base, env, algo)
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith('.json')]
        rewards = []
        for f in files:
            try:
                r = get_mean_reward(os.path.join(d, f))
                if r is not None:
                    rewards.append(r)
            except:
                pass
        if rewards:
            print(f'  {algo}: n={len(rewards)}, mean={np.mean(rewards):.1f} +/- {np.std(rewards):.1f}')
        else:
            print(f'  {algo}: 0 files')


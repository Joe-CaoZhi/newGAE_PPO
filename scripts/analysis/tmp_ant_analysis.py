import json
import numpy as np
import os


def load_final_rewards(path, n=5):
    rewards = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.json'):
            with open(os.path.join(path, fname)) as f:
                d = json.load(f)
            if 'eval_rewards' in d:
                r = d['eval_rewards']
            elif 'episode_rewards' in d:
                r = d['episode_rewards']
            else:
                keys = [k for k in d.keys() if 'reward' in k.lower()]
                r = d[keys[0]] if keys else []
            if r:
                rewards.append(np.mean(r[-n:]) if len(r) >= n else np.mean(r))
    return rewards

def load_all_timeseries(path):
    """Load the full reward time series for each seed"""
    all_series = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.json'):
            with open(os.path.join(path, fname)) as f:
                d = json.load(f)
            if 'eval_rewards' in d:
                r = d['eval_rewards']
            elif 'episode_rewards' in d:
                r = d['episode_rewards']
            else:
                keys = [k for k in d.keys() if 'reward' in k.lower()]
                r = d[keys[0]] if keys else []
            all_series.append(r)
    return all_series

base = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment/Ant-v4'
print("=== ICMLExperiment Ant-v4 Final Rewards ===")
for method in sorted(os.listdir(base)):
    path = os.path.join(base, method)
    if os.path.isdir(path):
        rewards = load_final_rewards(path)
        if rewards:
            print(f'{method}: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}, n={len(rewards)}, values={[round(r,1) for r in rewards]}')

print()
print("=== AntV3Validation Final Rewards ===")
base2 = '/Users/joe-caozhi/newGAE_ppo/results/AntV3Validation'
for method in sorted(os.listdir(base2)):
    path = os.path.join(base2, method)
    if os.path.isdir(path):
        rewards = load_final_rewards(path)
        if rewards:
            print(f'{method}: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}, n={len(rewards)}, values={[round(r,1) for r in rewards]}')

# Also check first JSON structure
print()
print("=== JSON key structure (Ant-v4 Optimal_PPO s0) ===")
p = '/Users/joe-caozhi/newGAE_ppo/results/ICMLExperiment/Ant-v4/Optimal_PPO/Optimal_PPO_s0.json'
with open(p) as f:
    d = json.load(f)
for k, v in d.items():
    if isinstance(v, list):
        print(f'  {k}: list of {len(v)} items, first={v[0] if v else None}, last={v[-1] if v else None}')
    else:
        print(f'  {k}: {v}')


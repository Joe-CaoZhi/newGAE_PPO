import json
import numpy as np
from pathlib import Path

BASE = Path('results/ICMLExperiment')

print('=== 完整实验数据汇总 ===')
envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate', 'Optimal_HCGAE_SCR']

for env in envs:
    print(f'\n=== {env} ===')
    for algo in algos:
        d = BASE / env / algo
        if d.exists():
            files = sorted(d.glob('*.json'))
            scores = []
            for f in files:
                try:
                    data = json.load(open(f))
                    if 'eval_rewards' in data and len(data['eval_rewards']) > 0:
                        scores.append(round(float(np.mean(data['eval_rewards'][-5:])), 1))
                except:
                    pass
            n = len(scores)
            if n > 0:
                print(f'  {algo}: n={n}, mean={np.mean(scores):.0f}+/-{np.std(scores):.0f}, seeds={[round(s) for s in scores]}')
            else:
                print(f'  {algo}: no valid data')
        else:
            print(f'  {algo}: dir not found')


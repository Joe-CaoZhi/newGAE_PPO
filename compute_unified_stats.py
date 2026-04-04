"""计算 UnifiedComparison 1M步数据的统计摘要"""
import json
from pathlib import Path

import numpy as np

ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
ALGOS = ['Standard_PPO', 'HCGAE_Imp12', 'PPO_KLPEN', 'PPO_Anneal', 'PPO_EntDecay', 'PPO_VClip']
SEEDS = [42, 123, 456, 789, 1234]

print("UnifiedComparison (1M步) 各算法最终性能统计 (均值 ± SEM, n=5种子)")
print("=" * 90)

for env in ENVS:
    print(f"\n{env}:")
    for algo in ALGOS:
        vals = []
        for s in SEEDS:
            fp = Path(f'results/UnifiedComparison/{env}/{algo}/{algo}_s{s}.json')
            if fp.exists():
                with open(fp) as f:
                    d = json.load(f)
                er = d.get('eval_rewards', [])
                v = float(np.mean(er[-5:])) if len(er) >= 5 else d.get('final_reward', 0)
                vals.append(v)
        if vals:
            m = np.mean(vals)
            sem = np.std(vals) / np.sqrt(len(vals))
            print(f"  {algo:<25}: {m:>8.0f} ± {sem:>5.0f}  (n={len(vals)})")
        else:
            print(f"  {algo:<25}: NO DATA")


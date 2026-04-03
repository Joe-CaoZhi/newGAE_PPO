#!/usr/bin/env python3
"""检查 DCPPO 多种子实验进度"""
import json
from pathlib import Path
import numpy as np

RESULTS_DIR = Path('/Users/joe-caozhi/newGAE_ppo/results/MultiEnv_DCPPO')
ENVS = ['Hopper-v4', 'Walker2d-v4']
SEEDS = [42, 123, 456, 789, 1234]
VARIANTS = ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']

print('\n=== Multi-Seed DCPPO 完成情况 ===')
all_done = True
for env in ENVS:
    print(f'\n[{env}]')
    for variant in VARIANTS:
        status = []
        rewards = []
        done_count = 0
        for seed in SEEDS:
            p = RESULTS_DIR / env / variant / f'{variant}_s{seed}_metrics.json'
            if p.exists():
                try:
                    d = json.load(open(p))
                    evr = d.get('eval_rewards', [])
                    n = len(evr)
                    if n >= 20:
                        final_r = float(np.mean(evr[-5:])) if len(evr) >= 5 else 0
                        status.append(f's{seed}✓({n}eval,r={final_r:.0f})')
                        rewards.append(final_r)
                        done_count += 1
                    else:
                        status.append(f's{seed}({n}/48eval)')
                        all_done = False
                except Exception as e:
                    status.append(f's{seed}(ERR:{e})')
                    all_done = False
            else:
                status.append(f's{seed}(miss)')
                all_done = False
        if rewards:
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            print(f'  {variant:15s}: {done_count}/5 done | mean={mean_r:.0f}±{std_r:.0f}')
        else:
            print(f'  {variant:15s}: 0/5 done')
        for s in status:
            print(f'    {s}')

print(f'\n{"All experiments DONE!" if all_done else "Experiments still running..."}')


#!/usr/bin/env python3
"""检查当前ICMLExperiment的完整结果"""
import json
import numpy as np
from pathlib import Path

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
algos_icml = ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']
algos_aligned = ['Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

print('=== ICMLExperiment Results (Final 5-eval mean) ===')
for env in envs:
    print(f'\n{env}:')
    for algo in algos_icml:
        scores = []
        for s in range(5):
            p = Path(f'results/ICMLExperiment/{env}/{algo}/{algo}_s{s}.json')
            if p.exists():
                d = json.load(open(p))
                evals = d.get('eval_rewards', [])
                if evals:
                    scores.append(np.mean(evals[-5:]))
        if scores:
            print(f'  {algo:40s}: {np.mean(scores):.1f} ± {np.std(scores)/max(np.sqrt(len(scores)),1):.1f}  (n={len(scores)})')
        else:
            print(f'  {algo:40s}: NO DATA')

print('\n\n=== AlignedExperiment Results ===')
aligned_dir = 'results/AlignedExperiment'
if not Path(aligned_dir).exists():
    print('  AlignedExperiment dir NOT FOUND')
    # check nearby dirs
    import os
    existing = [d for d in os.listdir('results') if 'ligned' in d or 'ICML' in d]
    print(f'  Available dirs with Aligned/ICML: {existing}')
else:
    for env in envs:
        print(f'\n{env}:')
        for algo in algos_aligned:
            scores = []
            for s in range(5):
                p = Path(f'{aligned_dir}/{env}/{algo}/{algo}_s{s}.json')
                if p.exists():
                    d = json.load(open(p))
                    evals = d.get('eval_rewards', [])
                    if evals:
                        scores.append(np.mean(evals[-5:]))
            if scores:
                print(f'  {algo:40s}: {np.mean(scores):.1f} ± {np.std(scores)/max(np.sqrt(len(scores)),1):.1f}  (n={len(scores)})')
            else:
                print(f'  {algo:40s}: NO DATA')

print('\n\nDone.')


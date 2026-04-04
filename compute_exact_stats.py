#!/usr/bin/env python3
"""Compute exact stats matching paper Table 1."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

ICML_DIR = Path('results/ICMLExperiment')
ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
ALGOS = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']

def load_scores(env, algo):
    scores = []
    for s in range(5):
        p = ICML_DIR / env / algo / f'{algo}_s{s}.json'
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            er = d.get('eval_rewards', [])
            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
            scores.append(final)
    return np.array(scores)

print('=== Table 1 (population std, ddof=0) ===')
all_data = {}
for env in ENVS:
    print(f'\n{env}:')
    all_data[env] = {}
    for algo in ALGOS:
        sc = load_scores(env, algo)
        all_data[env][algo] = sc
        print(f'  {algo:25s}: {np.mean(sc):6.0f} +/- {np.std(sc, ddof=0):4.0f}  (sample: +/-{np.std(sc, ddof=1):4.0f})')

print('\n\n=== Key Statistical Comparisons ===')
comparisons = [
    ('HCGAE vs Optimal PPO', 'Hopper-v4', 'Optimal_HCGAE', 'Optimal_PPO'),
    ('HCGAE vs Optimal PPO', 'Walker2d-v4', 'Optimal_HCGAE', 'Optimal_PPO'),
    ('HCGAE vs Optimal PPO', 'HalfCheetah-v4', 'Optimal_HCGAE', 'Optimal_PPO'),
    ('HCGAE vs Standard PPO', 'Hopper-v4', 'Optimal_HCGAE', 'Standard_PPO'),
    ('HCGAE vs Standard PPO', 'Walker2d-v4', 'Optimal_HCGAE', 'Standard_PPO'),
    ('HCGAE vs Standard PPO', 'HalfCheetah-v4', 'Optimal_HCGAE', 'Standard_PPO'),
    ('Optimal PPO vs Standard PPO', 'Hopper-v4', 'Optimal_PPO', 'Standard_PPO'),
    ('Optimal PPO vs Standard PPO', 'Walker2d-v4', 'Optimal_PPO', 'Standard_PPO'),
    ('Optimal PPO vs Standard PPO', 'HalfCheetah-v4', 'Optimal_PPO', 'Standard_PPO'),
]

for comp, env, algo1, algo2 in comparisons:
    a = all_data[env][algo1]
    b = all_data[env][algo2]
    delta_pct = (np.mean(a) - np.mean(b)) / np.mean(b) * 100
    u, p = mannwhitneyu(a, b, alternative='two-sided')

    # Cohen's d (pooled std, ddof=1)
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0

    p_bonf = min(p * 9, 1.0)
    sig = 'YES' if p_bonf < 0.05 else 'n.s.'
    env_short = env.replace('-v4', '')
    print(f'  {env_short:12s} {comp:30s}: Δ={delta_pct:+.1f}%, p={p:.3f}, p_bonf={p_bonf:.3f}, d={d:+.2f} [{sig}]')

print('\n=== AULC Sample Efficiency ===')
print('Area Under Learning Curve (mean trajectory integral, normalized by total steps)\n')
for env in ENVS:
    print(f'{env}:')
    for algo in ALGOS:
        curves = []
        for s in range(5):
            p = ICML_DIR / env / algo / f'{algo}_s{s}.json'
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                er = d.get('eval_rewards', [])
                es = d.get('eval_steps', [])
                if er and es:
                    curves.append((np.array(es), np.array(er)))

        if not curves:
            continue

        # Common grid
        max_step = min(c[0][-1] for c in curves)
        x_grid = np.linspace(0, max_step, 49)
        aulcs = []
        for steps, rewards in curves:
            y_interp = np.interp(x_grid, steps, rewards)
            aulc = np.trapz(y_interp, x_grid) / (x_grid[-1] - x_grid[0])
            aulcs.append(aulc)

        label = algo.replace('_', ' ')
        print(f'  {label:30s}: AULC = {np.mean(aulcs):6.0f} +/- {np.std(aulcs):4.0f}')


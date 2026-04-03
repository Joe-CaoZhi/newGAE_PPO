#!/usr/bin/env python3
"""
最终统计分析：计算所有环境的 SEM 和 Mann-Whitney 检验
输出更新到 paper_draft.md 需要的数字
"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

# ─── 加载数据 ──────────────────────────────────────────────────────────────
summary_path = Path('results/BaselineComparison/baseline_comparison_summary.json')
with open(summary_path) as f:
    data = json.load(f)

ENVS = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
METHODS = ['Standard_PPO', 'PPO_KLPEN', 'PPO_Anneal', 'PPO_EntDecay',
           'PPO_VClip', 'PPO_Full_Baseline', 'HCGAE_Imp12']
METHOD_LABELS = {
    'Standard_PPO': 'Standard PPO',
    'PPO_KLPEN': 'PPO-KLPEN',
    'PPO_Anneal': 'PPO-Anneal',
    'PPO_EntDecay': 'PPO-EntDecay',
    'PPO_VClip': 'PPO-VClip',
    'PPO_Full_Baseline': 'PPO-Full',
    'HCGAE_Imp12': 'HCGAE (Ours)',
}

print('='*80)
print('  FINAL STATISTICS FOR PAPER')
print('='*80)

# ─── Table 1: mean ± SEM ──────────────────────────────────────────────────
print('\n### Table 1: Performance (mean ± SEM, n=5, 300K steps)\n')
print(f"{'Method':<20} {'Hopper-v4':>16} {'Walker2d-v4':>16} {'HalfCheetah-v4':>18}")
print('-'*75)

for method in METHODS:
    row = f'{METHOD_LABELS[method]:<20}'
    for env in ENVS:
        d = data.get(env, {}).get(method, {})
        seeds = d.get('seeds', [])
        if seeds:
            m = np.mean(seeds)
            sem = np.std(seeds) / np.sqrt(len(seeds))
            row += f' {m:>7.0f} ±{sem:>4.0f}'
        else:
            row += f' {"N/A":>14}'
    print(row)

# ─── Mann-Whitney: HCGAE vs all baselines ─────────────────────────────────
print('\n### Mann-Whitney U tests: HCGAE vs. baselines\n')

for env in ENVS:
    print(f'\n--- {env} ---')
    hcgae_seeds = data[env].get('HCGAE_Imp12', {}).get('seeds', [])
    if not hcgae_seeds:
        print('  HCGAE data missing')
        continue

    print(f"{'Baseline':<20} {'U':>4} {'p-val':>8} {'Cohen\\'s d':>10} {'Sig':>4}")
    print('-'*55)

    for method in METHODS[:-1]:  # exclude HCGAE itself
        baseline_seeds = data[env].get(method, {}).get('seeds', [])
        if not baseline_seeds:
            continue

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(hcgae_seeds, baseline_seeds, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((np.std(hcgae_seeds)**2 + np.std(baseline_seeds)**2) / 2)
        d = (np.mean(hcgae_seeds) - np.mean(baseline_seeds)) / (pooled_std + 1e-9)

        # Significance stars
        if p < 0.001: sig = '***'
        elif p < 0.01: sig = '**'
        elif p < 0.05: sig = '*'
        elif p < 0.10: sig = '.'
        else: sig = 'n.s.'

        print(f'{METHOD_LABELS[method]:<20} {stat:>4.0f} {p:>8.3f} {d:>+10.2f} {sig:>4}')

# ─── DCPPO Summary (if available) ─────────────────────────────────────────
dcppo_path = Path('results/MultiEnv_DCPPO/dcppo_multiseed_summary.json')
if dcppo_path.exists():
    print('\n\n### DCPPO Multi-Seed Summary (from run_dcppo_multiseed.py)\n')
    with open(dcppo_path) as f:
        dcppo_data = json.load(f)

    for env in ['Hopper-v4', 'Walker2d-v4']:
        print(f'\n--- {env} ---')
        for variant in ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']:
            d = dcppo_data.get(env, {}).get(variant, {})
            m, s, n = d.get('mean'), d.get('std'), d.get('n_seeds', 0)
            seeds = d.get('seeds', [])
            if m is not None:
                sem = s / np.sqrt(n) if n > 1 else 0
                print(f'  {variant:<15}: {m:.0f} ± {sem:.0f} (SEM), ± {s:.0f} (std), n={n}')
                print(f'  Seeds: {[f"{x:.0f}" if x is not None else "miss" for x in seeds]}')
            else:
                print(f'  {variant:<15}: N/A (n={n})')

print('\n\n### Per-seed data for paper verification\n')
for env in ENVS:
    print(f'\n[{env}]')
    for method in METHODS:
        d = data[env].get(method, {})
        seeds = d.get('seeds', [])
        m = d.get('mean', 0)
        std = d.get('std', 0)
        sem = std / np.sqrt(len(seeds)) if seeds else 0
        print(f'  {METHOD_LABELS[method]:<20}: {[f"{x:.0f}" for x in seeds]} -> mean={m:.0f} sem={sem:.0f}')


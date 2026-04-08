#!/usr/bin/env python3
"""
最终统计分析：计算所有环境的 SEM 和 Mann-Whitney 检验
输出更新到 paper_draft.md 需要的数字
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

# Load data
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

# Table 1: mean +/- SEM
print('\n### Table 1: Performance (mean +/- SEM, n=5, 300K steps)\n')
header = "{:<20} {:>18} {:>18} {:>20}".format('Method', 'Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4')
print(header)
print('-'*80)

for method in METHODS:
    row = "{:<20}".format(METHOD_LABELS[method])
    for env in ENVS:
        d = data.get(env, {}).get(method, {})
        seeds = d.get('seeds', [])
        if seeds:
            m = np.mean(seeds)
            sem = np.std(seeds) / np.sqrt(len(seeds))
            cell = "{:>7.0f} +/-{:>4.0f}".format(m, sem)
        else:
            cell = "{:>14}".format('N/A')
        row += "  " + cell
    print(row)

# Mann-Whitney: HCGAE vs all baselines
print('\n### Mann-Whitney U tests: HCGAE vs. baselines\n')

for env in ENVS:
    print('\n--- {} ---'.format(env))
    hcgae_seeds = data[env].get('HCGAE_Imp12', {}).get('seeds', [])
    if not hcgae_seeds:
        print('  HCGAE data missing')
        continue

    print("{:<20} {:>4} {:>8} {:>10} {:>4}".format('Baseline', 'U', 'p-val', "Cohen's d", 'Sig'))
    print('-'*55)

    for method in METHODS[:-1]:  # exclude HCGAE itself
        baseline_seeds = data[env].get(method, {}).get('seeds', [])
        if not baseline_seeds:
            continue

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(hcgae_seeds, baseline_seeds, alternative='two-sided')

        # Cohen's d
        pooled_std = np.sqrt((np.std(hcgae_seeds)**2 + np.std(baseline_seeds)**2) / 2)
        cohens_d = (np.mean(hcgae_seeds) - np.mean(baseline_seeds)) / (pooled_std + 1e-9)

        # Significance stars
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        elif p < 0.10:
            sig = '.'
        else:
            sig = 'n.s.'

        label = METHOD_LABELS[method]
        print("{:<20} {:>4.0f} {:>8.3f} {:>+10.2f} {:>4}".format(label, stat, p, cohens_d, sig))

# DCPPO Summary (if available)
dcppo_path = Path('results/MultiEnv_DCPPO/dcppo_multiseed_summary.json')
if dcppo_path.exists():
    print('\n\n### DCPPO Multi-Seed Summary\n')
    with open(dcppo_path) as f:
        dcppo_data = json.load(f)

    for env in ['Hopper-v4', 'Walker2d-v4']:
        print('\n--- {} ---'.format(env))
        for variant in ['DCPPO_Base', 'DCPPO_ImpS', 'DCPPO_Full']:
            vd = dcppo_data.get(env, {}).get(variant, {})
            m = vd.get('mean')
            s = vd.get('std')
            n = vd.get('n_seeds', 0)
            seeds = vd.get('seeds', [])
            if m is not None:
                sem = s / np.sqrt(n) if n > 1 else 0
                seed_str = str([round(x, 0) if x is not None else None for x in seeds])
                print("  {:<15}: {:.0f} +/- {:.0f} (SEM), n={}".format(variant, m, sem, n))
                print("  Seeds: {}".format(seed_str))
            else:
                print("  {:<15}: N/A (n={})".format(variant, n))

print('\n\n### Per-seed data for verification\n')
for env in ENVS:
    print('\n[{}]'.format(env))
    for method in METHODS:
        d = data[env].get(method, {})
        seeds = d.get('seeds', [])
        m = d.get('mean', 0)
        std = d.get('std', 0)
        sem = std / np.sqrt(len(seeds)) if seeds else 0
        seed_vals = str([round(x, 0) for x in seeds])
        label = METHOD_LABELS[method]
        print("  {:<20}: {} -> mean={:.0f} sem={:.0f}".format(label, seed_vals, m, sem))


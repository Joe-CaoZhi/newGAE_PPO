#!/usr/bin/env python3
"""Check current experiment status and results."""
import glob
import json

import numpy as np

print("=== HalfCheetah: HCGAE vs HCGAE_v2 vs Optimal_PPO ===")
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']:
    files = sorted(glob.glob(f'results/ICMLExperiment/HalfCheetah-v4/{algo}/{algo}_s*.json'))
    if not files:
        print(f"  {algo}: no data")
        continue
    finals = []
    for f in files:
        d = json.load(open(f))
        er = d.get('eval_rewards', [])
        if er:
            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
            finals.append(final)
    if finals:
        print(f"  {algo} ({len(finals)} seeds): mean={np.mean(finals):.1f}, std={np.std(finals):.1f}, seeds={[f'{x:.0f}' for x in finals]}")

print("\n=== Hopper-v4 ===")
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']:
    files = sorted(glob.glob(f'results/ICMLExperiment/Hopper-v4/{algo}/{algo}_s*.json'))
    if not files:
        print(f"  {algo}: no data")
        continue
    finals = []
    for f in files:
        d = json.load(open(f))
        er = d.get('eval_rewards', [])
        if er:
            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
            finals.append(final)
    if finals:
        print(f"  {algo} ({len(finals)} seeds): mean={np.mean(finals):.1f}, std={np.std(finals):.1f}, seeds={[f'{x:.0f}' for x in finals]}")

print("\n=== Walker2d-v4 ===")
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']:
    files = sorted(glob.glob(f'results/ICMLExperiment/Walker2d-v4/{algo}/{algo}_s*.json'))
    if not files:
        print(f"  {algo}: no data")
        continue
    finals = []
    for f in files:
        d = json.load(open(f))
        er = d.get('eval_rewards', [])
        if er:
            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
            finals.append(final)
    if finals:
        print(f"  {algo} ({len(finals)} seeds): mean={np.mean(finals):.1f}, std={np.std(finals):.1f}, seeds={[f'{x:.0f}' for x in finals]}")

print("\n=== Ant-v4 ===")
for algo in ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR', 'Optimal_HCGAE_v2']:
    files = sorted(glob.glob(f'results/ICMLExperiment/Ant-v4/{algo}/{algo}_s*.json'))
    if not files:
        print(f"  {algo}: no data")
        continue
    finals = []
    for f in files:
        d = json.load(open(f))
        er = d.get('eval_rewards', [])
        if er:
            final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
            finals.append(final)
    if finals:
        print(f"  {algo} ({len(finals)} seeds): mean={np.mean(finals):.1f}, std={np.std(finals):.1f}, seeds={[f'{x:.0f}' for x in finals]}")

print("\n=== Missing data summary ===")
envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE', 'Optimal_HCGAE_SCR']
for env in envs:
    for algo in algos:
        files = glob.glob(f'results/ICMLExperiment/{env}/{algo}/{algo}_s*.json')
        n = len(files)
        if n < 5:
            print(f"  MISSING: {env}/{algo}: {n}/5 seeds")


#!/usr/bin/env python3
"""深度分析新旧实验差距原因 - 数学和物理层面"""
import json
from pathlib import Path

import numpy as np


def load_results(base_dir, env, algo, seeds=range(5)):
    results = []
    for s in seeds:
        p = Path(base_dir) / env / algo / f'{algo}_s{s}.json'
        if p.exists():
            d = json.load(open(p))
            evals = d.get('eval_rewards', d.get('eval_returns', []))
            ev_hist = d.get('ev_history', d.get('explained_variance', []))
            alpha_hist = d.get('alpha_history', [])
            if evals:
                final5 = np.mean(evals[-5:])
                results.append({
                    'seed': s,
                    'final5': final5,
                    'evals': evals,
                    'ev_history': ev_hist,
                    'alpha_history': alpha_hist,
                    'raw': d
                })
    return results

envs = ['Hopper-v4', 'Walker2d-v4', 'HalfCheetah-v4']
algos = ['Standard_PPO', 'Optimal_PPO', 'Optimal_HCGAE_v2', 'Optimal_HCGAE_v2_NoBdry', 'Optimal_HCGAE_v2_NoGate']

print("="*70)
print("=== 新旧实验结果对比（AlignedExperiment vs ICMLExperiment）===")
print("="*70)

for env in envs:
    print(f"\n{'='*50}")
    print(f"环境: {env}")
    print(f"{'='*50}")

    for algo in algos[:3]:
        # AlignedExperiment
        aligned = load_results('results/AlignedExperiment', env, algo)
        icml = load_results('results/ICMLExperiment', env, algo)

        if aligned:
            av = [r['final5'] for r in aligned]
            print(f"\n  [{algo}]")
            print(f"  AlignedExp (新): n={len(av)}, mean={np.mean(av):.1f}±{np.std(av)/np.sqrt(len(av)):.1f}(SEM)")
            print(f"    per-seed: {[round(v,1) for v in av]}")

        if icml:
            iv = [r['final5'] for r in icml]
            print(f"  ICMLExp  (旧): n={len(iv)}, mean={np.mean(iv):.1f}±{np.std(iv)/np.sqrt(len(iv)):.1f}(SEM)")
            print(f"    per-seed: {[round(v,1) for v in iv]}")

        if aligned and icml:
            delta = np.mean(av) - np.mean(iv)
            print(f"  Δ (新-旧): {delta:+.1f} ({delta/np.mean(iv)*100:+.1f}%)")

print("\n\n" + "="*70)
print("=== AlignedExperiment 消融组完整数据 ===")
print("="*70)

for env in envs:
    print(f"\n{env}:")
    for algo in algos:
        res = load_results('results/AlignedExperiment', env, algo)
        if res:
            vals = [r['final5'] for r in res]
            print(f"  {algo:<35}: n={len(vals)}, mean={np.mean(vals):.1f}±{np.std(vals)/np.sqrt(len(vals)):.1f}")

print("\n\n" + "="*70)
print("=== EV 轨迹分析（关键诊断）===")
print("="*70)

for env in envs:
    print(f"\n{env}:")
    for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
        res = load_results('results/AlignedExperiment', env, algo)
        for r in res[:2]:
            ev = r['ev_history']
            if ev and len(ev) > 5:
                ev_arr = np.array(ev)
                # EV at different training phases
                n = len(ev_arr)
                early = np.mean(ev_arr[:n//5]) if n > 5 else ev_arr[0]
                mid = np.mean(ev_arr[2*n//5:3*n//5]) if n > 5 else ev_arr[n//2]
                late = np.mean(ev_arr[-n//5:]) if n > 5 else ev_arr[-1]
                # EV growth rate in early phase
                if n > 10:
                    ev_deltas = np.diff(ev_arr[:n//5])
                    max_rate = np.max(ev_deltas) if len(ev_deltas) > 0 else 0
                    mean_rate = np.mean(ev_deltas) if len(ev_deltas) > 0 else 0
                else:
                    max_rate = 0
                    mean_rate = 0
                print(f"  {algo} s{r['seed']}: EV early={early:.3f}, mid={mid:.3f}, late={late:.3f}, "
                      f"max_rate={max_rate:.4f}, mean_rate={mean_rate:.4f}")

print("\n\n" + "="*70)
print("=== 关键问题：HalfCheetah 差距的来源 ===")
print("="*70)

env = 'HalfCheetah-v4'
for algo in ['Optimal_PPO', 'Optimal_HCGAE_v2']:
    aligned = load_results('results/AlignedExperiment', env, algo)
    icml = load_results('results/ICMLExperiment', env, algo)

    print(f"\n{algo}:")
    if aligned:
        print("  AlignedExp 各 seed 学习曲线（前5、中5、后5步评估均值）:")
        for r in aligned:
            evals = r['evals']
            if evals:
                n = len(evals)
                e1 = np.mean(evals[:5]) if n >= 5 else np.mean(evals)
                e2 = np.mean(evals[n//2-2:n//2+3]) if n >= 5 else np.mean(evals)
                e3 = np.mean(evals[-5:]) if n >= 5 else np.mean(evals)
                print(f"    s{r['seed']}: early={e1:.1f}, mid={e2:.1f}, final={e3:.1f}")
    if icml:
        print("  ICMLExp 各 seed 学习曲线:")
        for r in icml:
            evals = r['evals']
            if evals:
                n = len(evals)
                e1 = np.mean(evals[:5]) if n >= 5 else np.mean(evals)
                e2 = np.mean(evals[n//2-2:n//2+3]) if n >= 5 else np.mean(evals)
                e3 = np.mean(evals[-5:]) if n >= 5 else np.mean(evals)
                print(f"    s{r['seed']}: early={e1:.1f}, mid={e2:.1f}, final={e3:.1f}")


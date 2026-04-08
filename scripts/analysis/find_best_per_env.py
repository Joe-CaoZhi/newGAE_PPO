#!/usr/bin/env python3
"""Find the best-performing HCGAE variant per environment across ALL experiments."""
import json
import numpy as np
import os
from pathlib import Path

BASE = Path("results")

# Scan all result directories recursively
def find_all_results(base):
    results = {}  # (env, algo) -> list of final_rewards
    for root, dirs, files in os.walk(base):
        for f in files:
            if not f.endswith('.json') or 'summary' in f or 'analysis' in f:
                continue
            path = Path(root) / f
            try:
                d = json.load(open(path))
                env = d.get('env', '')
                agent = d.get('agent', '')
                if not env or not agent:
                    # Try to infer from path
                    parts = path.parts
                    for i, p in enumerate(parts):
                        if p in ['Hopper-v4','Walker2d-v4','HalfCheetah-v4','Ant-v4',
                                 'Hopper-v3','Walker2d-v3','HalfCheetah-v3','Ant-v3']:
                            env = p
                            if i+1 < len(parts):
                                agent = parts[i+1]
                            break
                if not env:
                    continue
                er = d.get('eval_rewards', [])
                if len(er) >= 3:
                    final = float(np.mean(er[-5:])) if len(er) >= 5 else float(np.mean(er))
                    key = (env, agent)
                    if key not in results:
                        results[key] = []
                    results[key].append(final)
            except Exception:
                pass
    return results

print("Scanning all experiment results...")
all_results = find_all_results(BASE)

# Group by environment
envs = sorted(set(e for e, a in all_results.keys()))
print(f"\n{'='*100}")
print(f"  BEST RESULTS PER ENVIRONMENT (all variants, all seeds)")
print(f"{'='*100}")

for env in envs:
    # Get all algo results for this env
    algo_stats = {}
    for (e, a), vals in all_results.items():
        if e != env:
            continue
        if len(vals) == 0:
            continue
        algo_stats[a] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'sem': float(np.std(vals)/max(len(vals)**0.5, 1)),
            'max': float(np.max(vals)),
            'n': len(vals),
            'vals': vals,
        }

    if not algo_stats:
        continue

    # Find baseline PPO
    ppo_mean = None
    for ppo_key in ['Optimal_PPO', 'Standard_PPO']:
        if ppo_key in algo_stats:
            ppo_mean = algo_stats[ppo_key]['mean']
            break

    # Sort by mean
    sorted_algos = sorted(algo_stats.items(), key=lambda x: -x[1]['mean'])

    print(f"\n{env}")
    print(f"  {'Algo':<35} {'Mean':>9} {'SEM':>7} {'Max':>9} {'Δ%':>8} {'n':>4}  {'Vals (rounded)'}")
    print(f"  {'-'*100}")
    for algo, st in sorted_algos:
        delta = f"{(st['mean']-ppo_mean)/(abs(ppo_mean)+1e-8)*100:+.1f}%" if ppo_mean else "—"
        vals_str = str([round(v) for v in sorted(st['vals'], reverse=True)[:6]])
        marker = " ★" if algo == sorted_algos[0][0] and 'HCGAE' in algo else ""
        ppo_marker = " [baseline]" if algo in ['Optimal_PPO', 'Standard_PPO'] else ""
        print(f"  {algo:<35} {st['mean']:>9.1f} {st['sem']:>7.1f} {st['max']:>9.1f} {delta:>8} {st['n']:>4}  {vals_str}{marker}{ppo_marker}")

print(f"\n{'='*100}")
print("★ = best HCGAE variant for this environment")
print("Δ% = vs Optimal_PPO (or Standard_PPO if Optimal_PPO not available)")


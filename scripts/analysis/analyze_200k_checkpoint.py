#!/usr/bin/env python3
"""分析ICMLExperiment在200K步时的检查点表现，和快速验证做对比"""
import json
from pathlib import Path

import numpy as np

ICML_DIR = Path("results/ICMLExperiment")
QUICK_DIR = Path("results/V4QuickValidation")
ENVS = ["Hopper-v4", "Walker2d-v4", "HalfCheetah-v4"]
ALGOS_ICML = ["Optimal_PPO", "Optimal_HCGAE_v2"]

print("="*70)
print("ICMLExperiment @ ~200K steps (rollout ~200K/2048 ≈ 98th rollout)")
print("="*70)

for env in ENVS:
    print(f"\n{env}:")
    for algo in ALGOS_ICML:
        vals_200k = []
        vals_500k = []
        for s in range(5):
            p = ICML_DIR / env / algo / f"{algo}_s{s}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            er = d.get('eval_rewards', [])
            es = d.get('eval_steps', [])
            if er and es:
                # find ~200K checkpoint
                idx_200k = None
                for i, step in enumerate(es):
                    if step >= 200000:
                        idx_200k = i
                        break
                if idx_200k is not None:
                    vals_200k.append(er[idx_200k])
                vals_500k.append(float(np.mean(er[-5:])))

        if vals_200k:
            m200 = np.mean(vals_200k)
            m500 = np.mean(vals_500k)
            sem200 = np.std(vals_200k)/max(np.sqrt(len(vals_200k)),1)
            print(f"  {algo}: @200K={m200:.0f}±{sem200:.0f}  @500K={m500:.0f}  (n={len(vals_200k)})")

print("\n")
print("="*70)
print("QuickValidation @ 200K steps (3 seeds)")
print("="*70)
for env in ENVS:
    print(f"\n{env}:")
    ppo_mean = None
    for algo in ["Optimal_PPO", "Optimal_HCGAE_v2", "Optimal_HCGAE_v4"]:
        vals = []
        for s in range(3):
            p = QUICK_DIR / env / algo / f"{algo}_s{s}.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            er = d.get('eval_rewards', [])
            if er:
                vals.append(float(np.mean(er[-5:])))
        if vals:
            m = np.mean(vals)
            sem = np.std(vals)/max(np.sqrt(len(vals)),1)
            delta = ""
            if algo != "Optimal_PPO" and ppo_mean is not None:
                delta = f"  Δ={100*(m-ppo_mean)/max(abs(ppo_mean),1):+.1f}%"
            print(f"  {algo}: {m:.0f}±{sem:.0f}  (n={len(vals)}){delta}")
            if algo == "Optimal_PPO":
                ppo_mean = m

print("\n")
print("="*70)
print("Conclusion: 200K steps is insufficient — methods need 300-500K to show benefits")
print("="*70)

